
DIRECTORY = 'yelp_dataset/'
BIZ_NAMES = ['business_id', 'name', 'address', 'city', 'state', \
    'postal code', 'latitude', 'longitude', 'stars', 'review_count', 'is_open', \
        'attributes', 'categories', 'hours']
client_id = 's3llWp81o_r38m0RSJnDxg'
API_key = 'X4h2OqoDVGBNi3t6wfHv5yYlhxQk_h4tBDWhTs7Fy2KRcJeJgExILpJCHZb67v23E9aKcJieCN1aMyP5NpB3rVWG3NMhKeAILOsxzMZgB2Ww_oq3J8m9km9DdWKJXHYx'

# # json.loads(line)
# all_reviews = {}
# with open(json_file_path) as fin:
#     for line in fin:
#         data = json.loads(line)
#         bizs = data['businesses']
#         print(data['total'])
#         print(len(bizs))
#         for biz in bizs:
#             alias = biz.get('alias').encode('utf-8').strip()
#             print(alias)
#             if alias:
#                 reviews =yelp_api.reviews_query(id=alias)
#                 all_reviews[alias] = reviews 
# json.dumps(all_reviews, DIRECTORY+'reviews_LA.json')

