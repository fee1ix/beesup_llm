OBSERVATION_ATTRIBUTES_DICT={
    'scientific_name':{
        'desc':'binary nomenclature consiting of genus and species, followed by taxonomic reference consisting of an authors name or abbreviation and a publication year.',
        #'desc':'Latin designation of the bee often consisting of genus and species. Mostly followed by th',
        'group':'bee',
        'suffix':' was observed'
    },
    # 'taxon':{
    #     'desc':'name of the author who first descriped the bee, and sometimes the respective publication year. In most cases the taxon is located rigth behind the latin designation of the bee',
    #     'group':'bee',
    #     'prefix':'(',
    #     'suffix':')',
    # },
    'date':{
        'sub_keys':['exact_date','date_range_orign', 'year_only']
    },
    'location':{
        'group':'location',
        'sub_keys':['nstate','fstate', 'district', 'near_city', 'loc', 'loc_desc', 'tk','coords_orign']
    },
    'n_males':{
        'desc':'number of observed male bees, could be abbreviated with "♂"-symbol after the respective amount',
        'group':'amounts',
        'suffix':'male'
    },
    'n_females':{
        'desc':'number of female bees/ worker bees, could be abbreviated with "♀"-symbol after the respective amount',
        'group':'amounts',
        'suffix':'female'
    },
    'n_queens':{
        'desc':'the number of queens is explicitly described',
        'group':'amounts',
        'suffix':'queen'

    },
    'n_divers':{
        'desc':'number of bees with unassignable gender, could be abbreviated with "☿"-symbol after the respective amount',
        'group':'amounts',
        'suffix':'divers gender'
    },
    'leg':{
        'desc':'person who confirmed the bee, could be abbreviated with "leg." in front of the respective name',
        'group':'persons',
        'prefix':'confirmed by',
        'suffix':'has confirmed'
    },
    'coll':{
        'desc':'person who collected the bee, could be abbreviated with "coll." in front of the respective name',
        'group':'persons',
        'prefix':'collected by',
        'suffix':'has collected'
    },
    'det':{
        'desc':'person who has determined the bee, could be abbreviated with "det." or "vid." in front of the respective name',
        'group':'persons',
        'prefix':'determined by',
        'suffix':'has determined'
    },
    # 'det_literature':{
    #     'desc':'literatur references used to determine the observation',
    #     'group':'source',
    #     'prefix':'determined by literature',
    #     'suffix':'was used for determination'
    # },
    'habitat':{
        'group':'location'
    },
    'visited_flowers':{
        'group':'location'
    },
    'behaviour':{
        'group':'bee'
    },
    'observed_nesting':{
        'group':'location'
    },
    'collecting_method':{
    },
    'remarks':{
        'desc':'additional notes or observations that do not fit into the other categories',
    },   
}

OBSERVATION_DESCRIPTIONS_DICT={k:OBSERVATION_ATTRIBUTES_DICT[k]['desc'] for k in OBSERVATION_ATTRIBUTES_DICT.keys() if 'desc' in OBSERVATION_ATTRIBUTES_DICT[k]}

FEW_SHOT_EXAMPLES=[
    {
        'report_passage':"""
Für das  Stadtgebiet Bielefeld nördlich des Teutoburger Waldes können für 
den  Zeitraum  1978-1996  73  Wildbienenarten  nachgewiesen  werden,  das 
entspricht etwa 30 Prozent der in NRW vorkommenden Arten.
Eine Rote Liste der Bienen liegt für Nordrhein-Westfalen bisher nicht vor, 
die  Rote  Liste  für  die  Bundesrepublik  basiert  auf Daten  aus  Süddeutsch
land und ist auf Nordrhein-Westfalen nur bedingt übertragbar, weshalb  im 
Rahmen dieser Arbeit auf Rote Listen kein Bezug genommen wird.
In  der  folgenden,  kommentierten  Artenliste  sind  alle  nachgewiesenen  Bie
nenarten mit  Fundort,  Funddatum und  einem  Hinweis  zur  Ökologie  (nach 
verschiedenen  Autoren)  aufgeführt.  Auf Oligolektie,  Polylektie  und  para- 
sitierende Arten wird unter den Punkten 7.3-7 5  eingegangen.

Andrena fulva (MÜLLER  1766)

Nachweise:  Bleichstraße  (1992),  Rochedale  Park  (1994,  1995), 
Parkanlage  "Weither  Str."  (1994),  viele ♀ bei Universität  (1994),  Twellbach 
(1996),  polylektische,  weit  verbreitet  Art,  diese  auffällige  Biene 
kann im Frühjahr überall im Stadtgebiet beobachtet werden
""".strip(),
        'gold_json':{
            "meta_scientific_name": "Andrena fulva (MÜLLER 1766)",
            "meta_location": "Nordrhein-Westfalen, Bielefeld nördlich des Teutoburger Waldes",
            "observations": 
                [
                    {
                        "date": "1992",
                        "location": "Bleichstraße"
                    },{
                        "date": "1994",
                        "location": "Rochedale Park"
                    },{
                        "date": "1995",
                        "location": "Rochedale Park"
                    },{
                        "date": "1994",
                        "location": "Parkanlage 'Weither Str.'"
                    },{
                        "date": "1994",
                        "location": "Universität",
                        "n_females": "viele"
                    },{
                        "date": "1996",
                        "location": "Twellbach"
                    },
                ]
            }

    },{
        'report_passage':"""
Beitr.Naturk.Niedersachsens  58  (2005):  2- 6

Hervorhebenswerte Stechimmenfunde aus dem ), Folge II östlichen Niedersachsen ( von Christian Helmreich und Reiner Theunert

Bei Erkundungen verschiedener Gebiete  im östlichen Niedersachsen wurden  Stechimmen 
nachgewiesen,  über  deren  Verbreitung  im  nordwestlichen  Deutschland  kaum  etwas  oder

2

nichts bekannt ist, so dass die Funde hervorhebenswert sind. Die nachfolgend verzeichneten 
Tiere befinden sich  in den  Sammlungen der Verfasser.  Zu jedem Fundort wird der TK 25- 
Quadrant mitgeteilt. Alle Funde wurden von Theunert gesammelt und bestimmt.

Andrena tarsata NYLANDER 1848; Apidae (Bienen)
Ortsrand  St.  Andreasberg  (4229/3);  2  Weibchen,  3.7.2000.  Nach  dem  2.  Weltkrieg  gab  es 
keine  Fundhinweise  mehr,  so  dass  die  Art  bisher  als  in  Niedersachsen  verschollen  galt 
(THEUNERT 2002).
Hylaeus variegatus (FABRICIUS 1798); Apidae (Bienen)
Bahnhof Walkenried (4429/2);  eine Königin, 3.8.2002. Den verschiedenen Fundangaben vor 
dem 2. Weltkrieg nach zu urteilen (vgl. THEUNERT  1994), dürfte die Art damals nicht so 
selten wie heute gewesen sein. In neuerer Zeit auch in Göttingen gefunden (BRAUN 1997).
Lasioglossum costulatum (KRIECHBAUMER 1873); Apidae (Bienen)
Heeseberg (3931/1);  1♂ 2 ♀,  6.8.2003.  Erstmals für Niedersachsen belegt.  Die Art ist 
angeblich einmal  1893  im hannoverschen Stadtwald Eilenriede gefunden worden (GEHRS 
1910). Die Angabe gilt als unsicher (THEUNERT  1994, DATHE & BLANK 2004), da ein 
Belegexemplar bis  heute  nicht  gefunden  wurde.  Andererseits jedoch  ist  die Art  kaum  zu 
verwechseln.
Megachile lagopoda (LINNAEUS 1761); Apidae (Bienen)
Braunkohlentagebau Helmstedt (3732/3);  3-4  Weibchen,  6.8.2003. Außer der Mitteilung von 
THEUNERT (2000) auf ein Vorkommen auf dem nur einige Kilometer entfernten Heeseberg 
stammen  alle  weiteren niedersächsischen Nachweise  aus  der Zeit um  1900  (THEUNERT 
2001). An den beiden Orten bei Helmstedt kommt auch die ähnliche Art Megachile maritima 
vor.
""".strip(),
        'gold_json':{
            "meta_location": "Deutschland, Niedersachsen",
            "meta_leg": "Theunert",
            "meta_det": "Theunert",
            "observations": 
                [
                    {
                        "scientific_name": "Andrena tarsata NYLANDER 1848",
                        "date": "3.7.2000",
                        "location": "Ortsrand  St.  Andreasberg  (4229/3)",
                        "n_females": "2"
                    },{
                        "scientific_name": "Hylaeus variegatus (FABRICIUS 1798)",
                        "date": "3.8.2002",
                        "location": "Bahnhof Walkenried (4429/2)",
                        "n_queens": "1"
                    },{
                        "scientific_name": "Lasioglossum costulatum (KRIECHBAUMER 1873)",   
                        "date": "6.8.2003",
                        "location": "Heeseberg (3931/1)",
                        "n_males": "1",
                        "n_females": "2"
                    },{
                        "scientific_name": "Megachile lagopoda (LINNAEUS 1761)",
                        "date": "6.8.2003",
                        "location": "Braunkohlentagebau Helmstedt (3732/3)",
                        "n_females": "3-4"
                    }
                ]
            }
    },{
        'report_passage':"""
Im Rahmen von Gutachten und Erfassungen konnten 
in  den  letzten  Jahren  einige  bemerkenswerte  Nach-
weise  von  Stechimmen  (Hymenoptera  Aculeata)  für 
Niedersachsen  und  die  eher  artenarme  Region  des 
nordwestdeutschen Flachlandes erbracht werden. Ein 
großer Teil  der  Daten  stammt  aus  Erfassungen  in  der 
Stadt Hannover, der Region Hannover sowie Beifänge 
aus dem Projekt „Hummelschutz in Niedersachsen“ des 
NABU Niedersachsen. Hier sollen vorab nur einige fau-
nistischen Besonderheiten publiziert werden. 
Weitere  Ergebnisse  stammen  von  kleineren  Einzel-
untersuchungen  (Helmstedt,  Uelzen)  sowie  anderen 
Aufsammlungen  und  ergänzenden  Meldungen  von 
Kollegen zu den Arten.

Lasioglossum costulatum (Kriechbaumer, 1873)

• Nienhagen/Weper an Campanula rotundifolia [51.7132° N 9.8013° E]   

1 ♀, 10.9.2016 (leg. / coll. Witt) 

Dritter  Fund  für  Niedersachsen.  Magerrasen  westlich 
des  Segelflugplatzgeländes  auf  dem  Höhenzug  der 
Weper. 
Der  Erstnachweis  aus  dem  Jahre  2003  stammt  vom 
Heeseberg  südlich  von  Helmstedt  (Theunert  2005). 
Theunert (2016) meldet einen aktuellen Fund aus dem 
Jahr 2014 aus Ehra-Lessien nördlich von Wolfsburg.
""".strip(),
        'gold_json':{
            "observations": 
                [
                    {
                        "scientific_name":"Lasioglossum costulatum (Kriechbaumer, 1873)",
                        "date": "10.9.2016",
                        "location": "Niedersachsen, Nienhagen/Weper, [51.7132°N, 9.8013° E]",
                        "n_females": "1",
                        "leg": "Witt",
                        "coll": "Witt",
                        "habitat": "Magerrasen",
                        "visited_flowers": "Campanula rotundifolia"
                    }
                ]
            }
    }
]

def build_extraction_prompt():
    extraction_prompt=""
    extraction_prompt+="""Your task is to extract data on wild bee observations described in the report passages into structured JSONs. The expected json format is specified below. \
Extract only the specified attributes. If an attribute of an observation is not specified or you can't find information for this attribute, leave it out. Don't make up information! Stick strictly to the information in the report passage. \
Extract information exactly as it appears in the passage."""

    extraction_prompt+="\n\nATTRIBUTE DESCRIPTIONS:\n"
    for k,v in OBSERVATION_DESCRIPTIONS_DICT.items():
        extraction_prompt+=f"{k}: {v}\n"
    
    extraction_prompt+="\nAll attributes in their defined order:\n"
    extraction_prompt+=", ".join(OBSERVATION_ATTRIBUTES_DICT.keys())


    extraction_prompt+="\n\nEXPECTED JSON FORMAT:\n"
    extraction_prompt+="""The JSON shall contain a list of observations as value for the key 'observations'. \
Each observation should be a dictionary containing the sorted attributes specified above as keys and observation-specific values extracted from the report passage. \
If an attribute is not specified for an observation, leave it out. \
If an attribute has the same value for all observations, it should be reported as meta-attribute with the prefix 'meta_'. \
Meta-attributes should be specified before 'observations' in the order defined above. \
If an attribute is specified as a meta-attribute, it should not be repeated in the observation-dictionary. \
Only 'location' can be specified as meta-attribute and observation-specific attribute at the same time. \
If you can't find any wild bee observations in the report passage return an empty JSON."""
    return extraction_prompt

EXTRACTION_PROMPT=build_extraction_prompt()


#PYDANTIC SCHEME
from pydantic import BaseModel#, ValidationError, Field
from typing import Optional, List

class ExtractionScheme4SingleObservation(BaseModel):
    scientific_name: Optional[str] = None
    date: Optional[str] = None
    location: Optional[str] = None
    n_males: Optional[str] = None
    n_females: Optional[str] = None
    n_divers: Optional[str] = None
    n_queens: Optional[str] = None
    leg: Optional[str] = None
    coll: Optional[str] = None
    det: Optional[str] = None
    habitat: Optional[str] = None
    visited_flowers: Optional[str] = None
    behaviour: Optional[str] = None
    observed_nesting: Optional[str] = None
    collecting_method: Optional[str] = None
    remarks: Optional[str] = None

    class Config:
        extra = 'ignore'

class ExtractionScheme4MultipeObservations(BaseModel):
    meta_scientific_name: Optional[str] = None
    meta_date: Optional[str] = None
    meta_location: Optional[str] = None

    meta_n_males: Optional[str] = None
    meta_n_females: Optional[str] = None
    meta_n_divers: Optional[str] = None
    meta_n_queens: Optional[str] = None

    meta_leg: Optional[str] = None
    meta_coll: Optional[str] = None
    meta_det: Optional[str] = None
    
    meta_habitat: Optional[str] = None
    meta_visited_flowers: Optional[str] = None
    meta_behaviour: Optional[str] = None
    meta_observed_nesting: Optional[str] = None
    meta_collecting_method: Optional[str] = None
    meta_remarks: Optional[str] = None
    observations: List[ExtractionScheme4SingleObservation] = None

    class Config:
        extra = 'ignore'

    
