import pandas as pd
import numpy as np
import re
import json

dataset = pd.read_json("data/Entity Recognition in Resumes.json", lines=True)
cv_text = np.array(dataset.content)

def clean_text(inp):
    def clean(txt):
        return "\n".join(
            line.replace("•", "").replace("-", "").replace("*", "").replace("#", " ") for line in txt.split("\n")
        )

    if isinstance(inp, list):
        return_out = ",".join([clean(string) for string in inp])
    elif isinstance(inp, str):
        return_out = clean(inp)

    return return_out


all_labels = []
for ind, annotation in enumerate(dataset.annotation):
    _ = [
        all_labels.append(entity_lst["label"])
        for entity_lst in annotation
        if entity_lst["label"] not in all_labels and len(entity_lst["label"]) > 0
    ]

dataset_reformatted = pd.DataFrame(
    columns=[
        "documentNum",
        "documentText",
        "documentAnnotation",
        "sentenceNum",
        "sentence",
        "labelsDict",
        "containsLabel",
        "wordNum",
        "word",
        "labelName",
    ]
)
k = 0

data_annotated = []
for df_index in range(len(dataset)):
    this_df = dataset.iloc[df_index]
    this_df_content = this_df['content']
    this_df_annotation = this_df["annotation"]
    this_df_annotation_sorted = sorted(this_df_annotation, key=lambda label_dict:label_dict['points'][0]['start'])
    is_begining = True
    is_last = False
    final_text_list = []
    final_out_list = []
    prev_end = None
    #i=0; label_dict = this_df_annotation_sorted[i]
    for i, label_dict in enumerate(this_df_annotation_sorted):
        if len(label_dict['label']) > 0:
            label = label_dict['label'][0]
            start = label_dict['points'][0]['start']
            end = label_dict['points'][0]['end']
            text = label_dict['points'][0]['text']
            if i == len(this_df_annotation_sorted)-1:
                is_last = True

            if is_begining:
                # Additional text i.e. Os
                extra_text = clean_text(this_df_content[:start])
                is_begining = False
            else:
                extra_text = clean_text(this_df_content[prev_end+1:start])

            prev_end = end
            text_list = [txt for txt in extra_text.split(' ') if txt.strip() != ""]
            final_text_list += text_list
            final_out_list += ["O"] * len(text_list)

            text_list = [txt for txt in clean_text(text).split(' ') if txt.strip() != ""]
            final_text_list += text_list
            final_out_list += [label.upper()+'-B']+([label.upper()+'-I']*(len(text_list)-1))

            if is_last:
                extra_text = clean_text(this_df_content[end + 1:])
                text_list = [txt for txt in extra_text.split(' ') if txt.strip() != ""]
                final_text_list += text_list
                final_out_list += ["O"] * len(text_list)

    data_annotated.append(tuple(zip(final_text_list, final_out_list)))

import pickle
with open('data/data_ready_list.pkl', 'wb') as out_file:
    pickle.dump(data_annotated, out_file)

for i in range(len(dataset)):
    dataset_reformatted.loc[k, "documentNum"] = i + 1
    dataset_reformatted.loc[k, "documentText"] = dataset.content[i]
    dataset_reformatted.loc[k, "documentAnnotation"] = dataset.annotation[i]
    skill_label = [
        cv_label["label"][0]
        for cv_label in dataset.annotation[i]
        if len(cv_label["label"]) > 0
    ]
    skills = [
        re.split(",\n", clean_text(cv_label["points"][0]["text"]))
        for cv_label in dataset.annotation[i]
        if len(cv_label["label"]) > 0
    ]
    skills_dict = dict(zip(skill_label, skills))
    dataset_reformatted.loc[k, "labelsDict"] = json.dumps(skills_dict)

    for sent_num, sent in enumerate(clean_text(dataset.content[i]).split("\n")):
        if sent.strip() != "":
            dataset_reformatted.loc[k, "sentenceNum"] = f"{i + 1}.{sent_num + 1}"
            dataset_reformatted.loc[k, "sentence"] = sent

            found_skills = dict()
            contains_labels = 0
            for it, val in skills_dict.items():
                ss = [in_val for in_val in val if in_val in sent]
                found_skills[it] = ss
                if len(ss) > 0:
                    contains_labels = 1

            dataset_reformatted.loc[k, "containsLabel"] = contains_labels

            for w, word in enumerate(sent.replace(",", "").split(" ")):
                dataset_reformatted.loc[
                    k, "wordNum"
                ] = f"{i + 1}.{sent_num + 1}.{w + 1}"
                dataset_reformatted.loc[k, "word"] = word

                if contains_labels:
                    for it, val in found_skills.items():
                        import pdb; pdb.set_trace()
                        if (
                            word in " ".join(val)
                            and " ".join(val).strip() != ""
                            and word.strip() != ""
                        ):
                            dataset_reformatted.loc[k, "labelName"] = it
                k += 1

dataset_reformatted.fillna("", inplace=True)
dataset_reformatted.loc[dataset_reformatted.labelName == "", "labelName"] = "O"
dataset_reformatted.to_pickle("../data/dataset_reformatted.pkl")

X_text_list = []
y_binary_list = []
y_ner_list = []
for sentenceNum in pd.unique(dataset_reformatted.sentenceNum):
    if sentenceNum != "":
        sent_data = dataset_reformatted[
            dataset_reformatted.wordNum.str.startswith(f"{sentenceNum}.")
        ]
        X_text_list.append(list(sent_data.word))
        y_ner_list.append(list(sent_data.labelName))
        if len([label for label in sent_data.labelName if label != "O"]) > 0:
            y_binary_list.append(1)
        else:
            y_binary_list.append(0)

dataset_ready = pd.DataFrame(
    data={"X_text": X_text_list, "y_binary": y_binary_list, "y_ner": y_ner_list}
)
dataset_ready.to_pickle("../data/dataset_ready.pkl")
