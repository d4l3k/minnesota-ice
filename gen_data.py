import json
import glob
import re
import motor.motor_asyncio
import dateutil.parser
import datetime
import argparse
import requests
import asyncio
import functools
import pymongo
from tqdm import tqdm
import numpy as np


mongo = motor.motor_asyncio.AsyncIOMotorClient().ice

year_re = re.compile("\d+")

num_days = 180

def load_json(fname):
  with open(fname) as f:
    return json.load(f)

def get_year(fname):
  return int(year_re.search(fname).group())


sites = {}

def deep_eq(_v1, _v2):
  """
  Tests for deep equality between two python data structures recursing
  into sub-structures if necessary. Works with all python types including
  iterators and generators. This function was dreampt up to test API responses
  but could be used for anything. Be careful. With deeply nested structures
  you may blow the stack.

  Doctests included:

  >>> x1, y1 = ({'a': 'b'}, {'a': 'b'})
  >>> deep_eq(x1, y1)
  True
  >>> x2, y2 = ({'a': 'b'}, {'b': 'a'})
  >>> deep_eq(x2, y2)
  False
  >>> x3, y3 = ({'a': {'b': 'c'}}, {'a': {'b': 'c'}})
  >>> deep_eq(x3, y3)
  True
  >>> x4, y4 = ({'c': 't', 'a': {'b': 'c'}}, {'a': {'b': 'n'}, 'c': 't'})
  >>> deep_eq(x4, y4)
  False
  >>> x5, y5 = ({'a': [1,2,3]}, {'a': [1,2,3]})
  >>> deep_eq(x5, y5)
  True
  >>> x6, y6 = ({'a': [1,'b',8]}, {'a': [2,'b',8]})
  >>> deep_eq(x6, y6)
  False
  >>> x7, y7 = ('a', 'a')
  >>> deep_eq(x7, y7)
  True
  >>> x8, y8 = (['p','n',['asdf']], ['p','n',['asdf']])
  >>> deep_eq(x8, y8)
  True
  >>> x9, y9 = (['p','n',['asdf',['omg']]], ['p', 'n', ['asdf',['nowai']]])
  >>> deep_eq(x9, y9)
  False
  >>> x10, y10 = (1, 2)
  >>> deep_eq(x10, y10)
  False
  >>> deep_eq((str(p) for p in xrange(10)), (str(p) for p in xrange(10)))
  True
  >>> str(deep_eq(range(4), range(4)))
  'True'
  >>> deep_eq(xrange(100), xrange(100))
  True
  >>> deep_eq(xrange(2), xrange(5))
  False
  """
  import operator, types

  def _deep_dict_eq(d1, d2):
    k1 = sorted(d1.keys())
    k2 = sorted(d2.keys())
    if k1 != k2: # keys should be exactly equal
      return False
    return sum(deep_eq(d1[k], d2[k]) for k in k1) == len(k1)

  def _deep_iter_eq(l1, l2):
    if len(l1) != len(l2):
      return False
    return sum(deep_eq(v1, v2) for v1, v2 in zip(l1, l2)) == len(l1)

  op = operator.eq
  c1, c2 = (_v1, _v2)

  # guard against strings because they are also iterable
  # and will consistently cause a RuntimeError (maximum recursion limit reached)
  for t in types.StringTypes:
    if isinstance(_v1, t):
      break
  else:
    if isinstance(_v1, types.DictType):
      op = _deep_dict_eq
    else:
      try:
        c1, c2 = (list(iter(_v1)), list(iter(_v2)))
      except TypeError:
        c1, c2 = _v1, _v2
      else:
        op = _deep_iter_eq

  return op(c1, c2)



async def load_water():
  await mongo.water.drop()
  await mongo.water.create_index('site')
  await mongo.water.create_index('time')
  await mongo.water.create_index([('time', pymongo.ASCENDING), ('site', pymongo.ASCENDING)])
  await mongo.water.create_index([('location', pymongo.GEOSPHERE)])
  await mongo.weather.create_index([('location', pymongo.GEOSPHERE), ('time', 1)])

  for fname in tqdm(glob.glob('data/water-*.json')):
    data = load_json(fname)
    for ts in data['value']['timeSeries']:
      site = ts['sourceInfo']
      name = site['siteName']
      sites[name] = site
      variable = ts['variable']['variableName']
      values = ts['values']
      if len(values) != 1:
        print('got non len 1 values field hmmm... {}, {}, {}'.format(name,
          variable, ts['variable']['options']))
        continue

      location = site['geoLocation']['geogLocation']

      operations = []

      for v in values[0]['value']:
        val = {
          variable: v['value'],
          'location': {
            'type': 'Point',
            'coordinates': [location['longitude'], location['latitude']],
          },
        }
        time = dateutil.parser.parse(v['dateTime'])
        key = {'site': name, 'time': time}
        val.update(key)
        operations.append(pymongo.UpdateOne(key, {'$set': val}, upsert=True))

      if len(operations) > 0:
        result = await mongo.water.bulk_write(operations)
        #print(result.bulk_api_result)

ice = []

def load_ice():
  global ice

  for fname in tqdm(glob.glob('data/ice-*.json'), desc="load ice"):
    ice.extend(load_json(fname)['results'])

  deduped = list(set([
    json.dumps(a) for a in ice
  ]))

  ice = [json.loads(a) for a in deduped]

  ice.sort(key=lambda x: x['ice_out_date'])

async def load_weather():
  #await mongo.stations.drop()
  #await mongo.stations.create_index([('location', pymongo.GEOSPHERE)])
  #await mongo.stations.create_index('name')

  #await mongo.weather.drop()
  #await mongo.weather.create_index([('location', pymongo.GEOSPHERE)])
  #await mongo.weather.create_index([('location', pymongo.GEOSPHERE), ('time', 1)])
  #await mongo.weather.create_index('time')

  r = requests.post('https://data.rcc-acis.org/StnMeta', data={
    'params': json.dumps({
      "state": "MN",
      "elems": ["maxt", "mint", "pcpn", "snow", "snwd"],
      "meta": ["ll", "name", "valid_daterange", "sids"]
    })
  })
  r.raise_for_status()
  stations = json.loads(r.content)
  stations = [
    station for station in stations['meta'] if 'll' in  station
  ]
  elems = [
    {"name": "maxt","interval": "dly","duration": "dly","add": "t"},
    {"name": "mint","interval": "dly","duration": "dly","add": "t"},
    {"name": "pcpn","interval": "dly","duration": "dly","add": "t"},
    {"name": "snow","interval": "dly","duration": "dly","add": "t"},
    {"name": "snwd","interval": "dly","duration": "dly","add": "t"}
  ]

  loop = asyncio.get_event_loop()
  reqs = [
    loop.run_in_executor(
      None,
      functools.partial(
        requests.post,
        'https://data.rcc-acis.org/StnData',
        data={
          "params": json.dumps({
            "sid": station['sids'][0],
            "sdate": "por",
            "edate": "por",
            "elems": elems
          })
        }
      )
    )
    for station in stations
  ]

  print('Requests made')

  wait = []

  for station, req in zip(stations, tqdm(reqs)):
    station['location'] = {
      'type': 'Point',
      'coordinates': station['ll'],
    }
    del station['ll']
    await mongo.stations.update({'name': station['name']}, {'$set': station}, upsert=True)

    r = await req
    r.raise_for_status()

    data = json.loads(r.content)['data']
    operations = []
    for datum in data:
      key = {
        'location': station['location'],
        'time': dateutil.parser.parse(datum[0])
      }
      val = {}
      val.update(key)
      for elem, v in zip(elems, datum[1:]):
        val[elem['name']] = v[0]

      operations.append(pymongo.UpdateOne(key, {'$set': val}, upsert=True))


    if len(operations) > 0:
      wait.append(mongo.weather.bulk_write(operations))

  await asyncio.wait(wait)

def date_range(start, end, step=datetime.timedelta(days=1)):
  dates = []
  while start <= end:
    dates.append(start)
    start += step
  return dates

def parse(n):
  try:
    return float(n)
  except ValueError:
    return -999999

async def get_lake_details(id):
    key = {"DOWNumber": id}
    lake = await mongo.lakes.find_one(key)
    if lake:
        return lake['result']

    loop = asyncio.get_event_loop()
    url = 'https://maps2.dnr.state.mn.us/cgi-bin/lakefinder/detail.cgi?type=lake_survey&id={}'.format(id)
    r = await loop.run_in_executor(None, requests.get, url)
    r.raise_for_status()

    body = json.loads(r.content)
    body['DOWNumber'] = id

    await mongo.lakes.update_one(key, {'$set': body}, upsert=True)

    return body['result']



async def process_ice_out(ice_out):
  day = ice_out['ice_out_date']
  end = dateutil.parser.parse(day)
  start = end - datetime.timedelta(days=num_days-1)
  days = date_range(start, end)
  assert len(days) == num_days

  lon = float(ice_out['lon'])
  lat = float(ice_out['lat'])
  location_q = {
    "$near": {
      "type": "Point",
      "coordinates": [lon, lat]
    }
  }

  # input data dimensions - (lon, lat, day, water speed, water temperature,
  # min temp, max temp, precipitation, snowfall, littoralAcres,
  # shoreLengthMiles, areaAcres, meanDepthFeet, maxDepthFeet,
  # averageWaterClarity)
  x = np.zeros((num_days, 15))
  x[:, :] = -999999

  x[:, 0] = lon
  x[:, 1] = lat
  x[:, 2] = [
    day.timetuple().tm_yday for day in days
  ]

  # output data dimensions - (days until ice out)
  y = np.array(range(num_days-1, -1, -1))

  streamflow = "Streamflow, ft&#179;/s"
  temp = "Temperature, water, &#176;C"

  reqs = [
    (
      mongo.weather.find_one({
        "location": location_q,
        "time": day,
        "snow": {"$ne": 'M'},
        }, {"mint": True, "maxt": True, "pcpn": True, "snow": True}),
      mongo.water.find_one({
        "location": location_q,
        streamflow: {"$exists": True},
        "time": day
      }, {streamflow: True}),
      mongo.water.find_one({
        "location": location_q,
        temp: {"$exists": True},
        "time": day
      }, {temp: True})
    )
    for day in days
  ]

  details = await get_lake_details(ice_out['id'])
  if details:
    x[:, 9] = details['littoralAcres']
    x[:, 10] = details['shoreLengthMiles']
    x[:, 11] = details['areaAcres']
    x[:, 12] = details['meanDepthFeet']
    x[:, 13] = details['maxDepthFeet']
    x[:, 14] = parse(details['averageWaterClarity'])


  for i, req in enumerate(reqs):
    stream = await req[1]
    if stream is not None:
      x[i, 3] = parse(stream[streamflow])

    watertemp = await req[2]
    if watertemp is not None:
      x[i, 4] = parse(watertemp[temp])

    weather = await req[0]
    if weather is not None:
      x[i, 5] = parse(weather["mint"])
      x[i, 6] = parse(weather["maxt"])
      x[i, 7] = parse(weather["pcpn"])
      x[i, 8] = parse(weather["snow"])

  return (x, y)


  """
  weather = mongo.weather.aggregate([
    {
      "$geoNear": {
        "near": {
          "type": "Point",
          "coordinates": [float(ice_out['lon']), float(ice_out['lat'])]
        },
        "distanceField": "distance",
        "spherical": True,
        "limit": 10000000000,
        "query": {
          "time": {"$gt": start, "$lte": end},
          "snow": {"$ne": 'M'},
        }
      }
    },
    {
      "$group": {
        "_id": "$time",
        "weather": {"$first": "$$ROOT"}
      }
    },
    {"$sort": {"_id": -1}}
  ])
  print(await weather.explain())

  async for day in weather:
    print(day)
  """


async def main():
  print('Running')

  parser = argparse.ArgumentParser()
  parser.add_argument("-water", action="store_true")
  parser.add_argument("-weather", action="store_true")
  parser.add_argument("-gen", action='store_true')
  parser.add_argument("-repl", action='store_true')
  args = parser.parse_args()

  if args.water:
    await load_water()
  if args.weather:
    await load_weather()
  if args.gen:
    load_ice()

    X = []
    Y = []


    queued = []
    async def pop():
      x, y = await queued.pop()
      X.append(x)
      Y.append(y)

    for ice_out in tqdm(ice, desc="process ice"):
      queued.append(process_ice_out(ice_out))

      if len(queued) >= 16:
        await pop()

    while len(queued) > 0:
      await pop()

    np.savez('data3.npz', x=X, y=Y)

  if args.repl:
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
  loop = asyncio.get_event_loop()
  loop.run_until_complete(main())

