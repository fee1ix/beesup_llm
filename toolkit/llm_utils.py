import pandas as pd


def get_inference_df(inference_outputs, tokenizer=None):

    inference_data=[]
    for i in range(len(inference_outputs['all_input_ids'])):
        row=dict()

        for col in ['all_input_ids', 'all_label_ids', 'all_all_ids', 'all_losses']:
            if isinstance(inference_outputs[col],type(None)): continue

            row[col[4:]]=inference_outputs[col][i]
        
        inference_data.append(row)

    inference_df=pd.DataFrame(inference_data)
    
    return inference_df