
DIRECTORY = 'yelp_dataset/'
BIZ_NAMES = ['address', 'attributes', 'business_id', 'categories', 'city', 'hours', 'is_open', 'latitude', 'longitude', 'name', 'postal_code', 'review_count', 'stars', 'state']

REV_NAMES = ['review_id', 'user_id', 'business_id', 'stars', 'date', 'text', 'useful', 'funny', 'cool']
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

