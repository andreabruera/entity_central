from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import re

def clean_list(wrong_list):
    
    good_list = []
    for w in wrong_list:
        if w not in good_list:
            good_list.append(w)
    return good_list

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

# Querying for: person, UK, film actor, with filmography

sparql.setQuery("""
SELECT ?item ?itemLabel
WHERE
{
    ?item wdt:P31 wd:Q5;
    wdt:P1283 ?filmography.
    ?item wdt:P27 wd:Q145.
    ?item wdt:P106 wd:Q10800557.
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
""")


sparql.setReturnFormat(JSON)
results = sparql.query().convert()
results_df = pd.json_normalize(results['results']['bindings'])

actors = clean_list([(re.sub('.+\/', '', a), b) for a, b in zip(results_df['item.value'].to_list(), results_df['itemLabel.value'].to_list())])

# Querying for: person, UK, football player, who received the English Football Hall of Fame

sparql.setQuery("""
SELECT ?item ?itemLabel
WHERE
{
    ?item wdt:P31 wd:Q5.
    ?item wdt:P27 wd:Q145.
    ?item wdt:P106 wd:Q937857.
    ?item wdt:P166 wd:Q1323117.
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
""")

sparql.setReturnFormat(JSON)
results = sparql.query().convert()
results_df = pd.json_normalize(results['results']['bindings'])

footballers = clean_list([(re.sub('.+\/', '', a), b) for a, b in zip(results_df['item.value'].to_list(), results_df['itemLabel.value'].to_list())])

# Querying for: person, UK, writer, with notable works, member of royal society of literature.

sparql.setQuery("""
SELECT ?item ?itemLabel
WHERE
{
    ?item wdt:P31 wd:Q5;
    wdt:P27 wd:Q145;
    wdt:P106 wd:Q6625963;
    wdt:P463 wd:Q1468277;
    wdt:P800 ?notable.
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
""")


sparql.setReturnFormat(JSON)
results = sparql.query().convert()
results_df = pd.json_normalize(results['results']['bindings'])

writers = clean_list([(re.sub('.+\/', '', a), b) for a, b in zip(results_df['item.value'].to_list(), results_df['itemLabel.value'].to_list())])

# Querying for: person, UK, politician, leader of the opposition

sparql.setQuery("""
SELECT ?item ?itemLabel
WHERE
{
    ?item wdt:P31 wd:Q5;
    wdt:P27 wd:Q145;
    wdt:P106 wd:Q82955;
    wdt:P39 wd:Q2741536.
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
""")


sparql.setReturnFormat(JSON)
results = sparql.query().convert()
results_df = pd.json_normalize(results['results']['bindings'])

leaders_opposition = clean_list([(re.sub('.+\/', '', a), b) for a, b in zip(results_df['item.value'].to_list(), results_df['itemLabel.value'].to_list())])

# Querying for: person, UK, politician, prime minister

sparql.setQuery("""
SELECT ?item ?itemLabel
WHERE
{
    ?item wdt:P31 wd:Q5;
    wdt:P27 wd:Q145;
    wdt:P106 wd:Q82955;
    wdt:P39 wd:Q14211.
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
""")


sparql.setReturnFormat(JSON)
results = sparql.query().convert()
results_df = pd.json_normalize(results['results']['bindings'])

prime_ministers = clean_list([(re.sub('.+\/', '', a), b) for a, b in zip(results_df['item.value'].to_list(), results_df['itemLabel.value'].to_list())])

politicians = clean_list(leaders_opposition + prime_ministers)

# Querying for: tourist attraction, UK, declared of national interest

sparql.setQuery("""
SELECT ?item ?itemLabel
WHERE
{
    ?item wdt:P31 wd:Q570116;
    wdt:P17 wd:Q145;
    wdt:P1435 wd:Q219538.
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
""")


sparql.setReturnFormat(JSON)
results = sparql.query().convert()
results_df = pd.json_normalize(results['results']['bindings'])

attractions = clean_list([(re.sub('.+\/', '', a), b) for a, b in zip(results_df['item.value'].to_list(), results_df['itemLabel.value'].to_list())])

# Querying for: city, UK

sparql.setQuery("""
SELECT ?item ?itemLabel
WHERE
{
    ?item wdt:P31 wd:Q515;
    wdt:P17 wd:Q145.
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
""")


sparql.setReturnFormat(JSON)
results = sparql.query().convert()
results_df = pd.json_normalize(results['results']['bindings'])

cities = clean_list([(re.sub('.+\/', '', a), b) for a, b in zip(results_df['item.value'].to_list(), results_df['itemLabel.value'].to_list())])

import pdb; pdb.set_trace()
