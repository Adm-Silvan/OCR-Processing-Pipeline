Initial query result for all swiss humans: 
69294



SELECT ?obj ?name ?description ?first_name ?last_name ?birthdate ?deathdate 
(GROUP_CONCAT(distinct ?occupation_title;separator="; ") as ?occupations) 
(GROUP_CONCAT(distinct ?position_title;separator="; ") as ?positions) 
WHERE  {
?obj wdt:P31 wd:Q5;
     rdfs:label ?name;
     wdt:P27 wd:Q39;
     wdt:P569 ?birthdate;
     OPTIONAL{?obj schema:description ?description. FILTER(LANG(?description) = "de")}
     OPTIONAL{?obj wdt:P570 ?deathdate.}
     OPTIONAL{?obj wdt:P106 ?occupation. ?occupation rdfs:label ?occupation_title. FILTER(LANG(?occupation_title) = "de")}
     OPTIONAL{?obj wdt:P39 ?position. ?position rdfs:label ?position_title. FILTER(LANG(?position_title) = "de")}
     OPTIONAL{?obj wdt:P735 ?first_name. }
     OPTIONAL{?obj wdt:P734 ?last_name. }
     FILTER(LANG(?name) = "de")
}
GROUP BY ?obj ?name ?description ?first_name ?last_name ?birthdate ?deathdate
LIMIT 100