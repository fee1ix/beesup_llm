import pandas as pd

TOC="""
Wildbienen
├──  Nist- und Bauverhalten von Wildbienen
│   ├──  Familienübersicht der Wildbienen
│   ├──  Pollensammelverhalten: Vielfalt und Präferenzen
│   ├──  Nistplatzwahl und Nistplatztypen
│   ├──  Sozialverhalten von Wildbienen
│   └──  Baumaterialien und Requisiten für die Nist- und Bauweise von Wildbienen
└──  Umgangssprachliche Bezeichnungen der Wildbienen
    ├──  Lebensweise und Merkmale
    │   ├──  Lebensweise und Nestbau
    │   ├──  Merkmale der Wildbienen
    │   ├──  Vorkommen und Lebensweise
    │   └──  Verbreitung und Lebensraum
    └──  Vorkommen und Lokalitäten der Dünen-Steppenbiene
        ├──  Umgangssprachliche Bezeichnungen der Wildbienen-Arten
        ├──  Kuckuckswespen (Nomaden)
        └──  Wildbienenarten der Dünen-Steppenbiene
""".strip()


def get_system_prompt(toc=None, rag=False, **kwargs):
    prompt=""
    prompt+="""
You are a knowledge-oriented AI assistant specialised in wild bee knowledge. \
Provide only factually accurate information. \
If you are uncertain about an answer, state that you do not know rather than guessing or making assumptions.
""".strip()
    
    if toc:
        prompt+="\n\nYour wild-bee knowledge is organized in the following knowledge-tree:\n"
        prompt+=f"{toc}"
    
    if rag:
       prompt+="\n\n"
       prompt+="""
You have access to relevant knowledge chunks. \
Your goal is to answer the user's question strictly based on the provided context. \
If the answer is not present in the context, state "I don't know". \
Do not attempt to answer using outside knowledge.
""".strip()

    return prompt


def get_context(briefing_df=pd.DataFrame(), chunk_col='chunk',**kwargs):

    #RAG BRIEFING if briefing_df is passed
    context=""
    if not briefing_df.empty:
        chunk_col=kwargs.get('chunk_col', 'chunk')
        context+="\n\n### CONTEXT:\n\n"
        for _,row in briefing_df.iterrows():
            context+=f"{row[chunk_col]}\n\n"

    return context

def get_mcq_prompt(sample, briefing_df=pd.DataFrame(), **kwargs):

        prompt=""
        prompt+="""
You are provided with the following multiple-choice question. \
Carefully review the options and select the correct one. \
Respond only with the letter of the correct choice. \
Do not provide any additional explanation or reasoning.
""".strip()
        
        #RAG BRIEFING if briefing_df is passed
        prompt+=get_context(briefing_df, **kwargs)

        prompt+=f"\n\n### QUESTION:\n\n"
        prompt+=f"{sample.question}\n\n"

        for i, choice in enumerate(sample.choices):
            prompt+=f"{chr(65+i)}) {choice}\n"

        prompt+="\n### LETTER OF CORRECT CHOICE:\n"

def get_qdq_prompt(sample, fewshots_df=pd.DataFrame(), briefing_df=pd.DataFrame(), **kwargs):

        prompt=""
        prompt+="You are given a question in German that asks for a set of wild bee species meeting specific characteristics. "
        prompt+="Your answer should consist of a semicolon-separated list of scientific names of the wild bees. "
        prompt+="A scientific name must always follow the format: <genus> <species>. "
        prompt+="Do not generate information that is not verifiable! "
        prompt+="Only include wild bee species that you are absolutely certain match the described characteristics. "
        prompt+="If you are unsure or do not know the answer, please respond with 'I do not know'. "

        if not fewshots_df.empty:
            prompt+="\n\n### EXAMPLES:\n\n"

            for _,fewshot in fewshots_df.iterrows():
                prompt+=f"QUESTION: {fewshot.question} "
                prompt+=f"ANSWER: {'; '.join(fewshot.gold_items)}\n\n"

        #RAG BRIEFING if briefing_df is passed
        prompt+=get_context(briefing_df, **kwargs)

        prompt+=f"\n\n### QUESTION: "
        prompt+=f"{sample.question}\n"
        prompt+="\n### ANSWER: "

             
def get_ffq_prompt(sample, briefing_df,**kwargs):
     
    prompt=""
    prompt+=get_context(briefing_df, **kwargs)
    prompt+="\n\n### QUESTION:\n"
    prompt+=f"{sample.question}\n"

    return prompt
     

