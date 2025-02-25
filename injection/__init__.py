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


def get_system_prompt(toc=None, rag=False, briefing_df=pd.DataFrame(), **kwargs):
    """
    Returns the system prompt message for the wild bee AI assistant.

    Args:
    toc: str, optional
        Table of contents for the knowledge-tree.
    
    briefing_df: pd.DataFrame, optional
        DataFrame containing the briefing context, if passed non empty, assume RAG System.
    
    rag: bool, optional
        If True, assume RAG System.
    """

    prompt=""
    prompt+="You are a knowledge-oriented AI assistant specialised in wild bee knowledge. "
    prompt+="Provide only factually accurate information. "
    prompt+="If you are uncertain about an answer, state that you do not know rather than guessing or making assumptions. "
    prompt=prompt.strip()
    prompt+="\n\n"
    
    if toc:
        prompt+=f"Your wild-bee knowledge is organized in the following knowledge-tree:\n"
        prompt+=f"{toc}"
        prompt+="\n\n"
    
    if (not briefing_df.empty) or rag:
        prompt+="You have access to relevant knowledge chunks. "
        prompt+="Your goal is to answer the user's question strictly based on the provided context. "
        prompt+="If the answer is not present in the context, state \"I don't know\". "
        prompt+="Do not attempt to answer using outside knowledge. "
        prompt=prompt.strip()
       
    return prompt.strip()


def get_context(briefing_df=pd.DataFrame(), chunk_col='chunk', **kwargs):

    #RAG BRIEFING if briefing_df is passed
    context=""
    if not briefing_df.empty:

        chunk_col=kwargs.get('chunk_col', 'chunk')
        assert chunk_col in briefing_df, f"{chunk_col} missing in briefing_df"

        context+="### CONTEXT:\n\n"
        for _,row in briefing_df.iterrows():
            context+=f"{row[chunk_col]}\n\n"

    return context


def get_mcq_prompt(question, choices, briefing_df=pd.DataFrame(), **kwargs):

    prompt=""
    prompt+="You are provided with the following multiple-choice question. "
    prompt+="Carefully review the options and select the correct one. "
    prompt+="Respond only with the letter of the correct choice. "
    prompt+="Do not provide any additional explanation or reasoning. "
    prompt=prompt.strip()

    #RAG BRIEFING if briefing_df is passed
    prompt+=get_context(briefing_df, **kwargs)

    prompt+=f"\n\n### QUESTION:\n\n"
    prompt+=f"{question}\n\n"

    for i, choice in enumerate(choices):
        prompt+=f"{chr(65+i)}) {choice}\n"

    prompt+="\n### LETTER OF CORRECT CHOICE:\n"

    return prompt

def get_qdq_prompt(question, fewshots_df=pd.DataFrame(), briefing_df=pd.DataFrame(), **kwargs):

    prompt=""
    prompt+="You are given a question in German that asks for a set of wild bee species meeting specific characteristics. "
    prompt+="Your answer should consist of a semicolon-separated list of scientific names of the wild bees. "
    prompt+="A scientific name must always follow the format: <genus> <species>. "
    prompt+="Do not generate information that is not verifiable! "
    prompt+="Only include wild bee species that you are absolutely certain match the described characteristics. "
    #prompt+="If you are unsure or do not know the answer, please respond with \"I don't know\". "
    prompt=prompt.strip()

    if not fewshots_df.empty:

        assert 'question' in fewshots_df, "question missing in fewshots_df"
        assert 'gold_items' in fewshots_df, "gold_items missing in fewshots_df"

        prompt+="\n\n### EXAMPLES:\n\n"

        for _,fewshot in fewshots_df.iterrows():
            prompt+=f"QUESTION: {fewshot.question} "
            prompt+=f"ANSWER: {'; '.join(fewshot.gold_items)}\n\n"

    prompt+="\n"

    #RAG BRIEFING if briefing_df is passed
    prompt+=get_context(briefing_df, **kwargs)

    prompt+=f"\n\n### QUESTION: "
    prompt+=f"{question}\n"
    prompt+="\n### ANSWER: "
    return prompt
            
def get_ffq_prompt(question, briefing_df=pd.DataFrame(), **kwargs):
     
    prompt=""
    if not briefing_df.empty:
        prompt+=get_context(briefing_df, **kwargs)
        prompt+="### QUESTION:\n"

    prompt+=f"{question}"
    return prompt
     

