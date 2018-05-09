package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/pkg/errors"
)

const (
	waterURL     = `https://waterservices.usgs.gov/nwis/dv/?format=json&stateCd=mn&startDT=%d-01-01&endDT=%d-12-31&siteStatus=active`
	iceURL       = `https://maps1.dnr.state.mn.us/cgi-bin/climatology/ice_out_by_year.cgi?year=%d`
	satelliteURL = `http://ge.ssec.wisc.edu/modis-today/index.php?gis=true&filename=t1_07297_USA3_721_250m&date=2007_10_24_297&product=false_color&resolution=250m&overlay_sector=false&overlay_state=true&overlay_coastline=false`
)

var (
	start    = flag.Int("start", 1918, "year to start")
	end      = flag.Int("end", 2018, "year to end")
	category = flag.String("cat", "water", "ice,water")
)

func fetch(year int) error {
	log.Printf("Fetching %d...", year)
	var fetch string
	switch *category {
	case "water":
		fetch = fmt.Sprintf(waterURL, year, year)
	case "ice":
		fetch = fmt.Sprintf(iceURL, year)
	default:
		return errors.Errorf("unknown category: %q", category)
	}
	resp, err := http.Get(fetch)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return errors.Errorf("expected 200, got %d", resp.StatusCode)
	}
	f, err := os.OpenFile(fmt.Sprintf(`data/%s-%d.json`, *category, year), os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := io.Copy(f, resp.Body); err != nil {
		return err
	}
	return nil
}

func main() {
	flag.Parse()

	log.Printf("category: %q", *cat)

	for year := *end; year >= *start; year-- {
		if err := fetch(year); err != nil {
			log.Fatalf("%+v", err)
		}
	}
}
