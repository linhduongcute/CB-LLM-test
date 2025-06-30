import concepts

example_name = {'SetFit/sst2': 'text', 'ag_news': 'text', 'yelp_polarity': 'text', 'dbpedia_14': 'content', "TimSchopf/medical_abstracts": 'medical_abstract', "darklord1611/legal_citations": 'case_text', "darklord1611/ecom_categories": 'text'} # Done
concepts_from_labels = {'SetFit/sst2': ["negative","positive"], 'yelp_polarity': ["negative","positive"], 'ag_news': ["world", "sports", "business", "technology"], 'dbpedia_14': ["company","education","artist","athlete","office","transportation","building","natural","village","animal","plant","album","film","written"], "TimSchopf/medical_abstracts": ["neoplasms", "digestive_system_diseases", "nervous_system_diseases", "cardiovascular_diseases", "general_pathological_diseases"], "darklord1611/legal_citations": ["affirmed", "applied", "approved", "cited", "considered", "discussed", "distinguished", "followed", "referred to", "related"], "darklord1611/ecom_categories": ["household", "books", "electronics", "clothing_accessories"]} # Done

class_num = {'SetFit/sst2': 2, 'ag_news': 4, 'yelp_polarity': 2, 'dbpedia_14': 14, "TimSchopf/medical_abstracts": 5, "darklord1611/legal_citations": 10, "darklord1611/ecom_categories": 4} # Done

# Config for Roberta-Base baseline
finetune_epoch = {'SetFit/sst2': 3, 'ag_news': 2, 'yelp_polarity': 2, 'dbpedia_14': 2}
finetune_mlp_epoch = {'SetFit/sst2': 30, 'ag_news': 5, 'yelp_polarity': 3, 'dbpedia_14': 3}

# Config for CBM training
concept_set = {'SetFit/sst2': concepts.sst2, 'yelp_polarity': concepts.yelpp, 'ag_news': concepts.agnews, 'dbpedia_14': concepts.dbpedia, "TimSchopf/medical_abstracts": concepts.med_abs, "darklord1611/legal_citations": concepts.legal, "darklord1611/ecom_categories": concepts.ecom}


cbl_epochs = {'SetFit/sst2': 10, 'ag_news': 3, 'yelp_polarity': 2, 'dbpedia_14': 2, "TimSchopf/medical_abstracts": 10, "darklord1611/legal_citations": 10, "darklord1611/ecom_categories": 10}

dataset_config = {
    "TimSchopf/medical_abstracts": {
        "text_column": "medical_abstract",
        "label_column": "condition_label"
    },
    "SetFit/20_newsgroups": {
        "text_column": "text",
        "label_column": "label"
    },
    "JuliaTsk/yahoo-answers": {
        "text_column": "question title",
        "label_column": "class id"
    },
    "fancyzhx/ag_news": {
        "text_column": "text",
        "label_column": "label"
    },
    "fancyzhx/dbpedia_14": {
        "text_column": "content",
        "label_column": "label"
    },
    "SetFit/sst2": {
        "text_column": "text",
        "label_column": "label"
    },
    "fancyzhx/yelp_polarity": {
        "text_column": "text",
        "label_column": "label"
    },
    "pietrolesci/pubmed-200k-rct": {
        "text_column": "text",
        "label_column": "labels"
    },
    "dd-n-kk/uci-drug-review-cleaned": {
        "text_column": "review",
        "label_column": "rating"
    },
    "darklord1611/legal_citations": {
        "text_column": "case_text",
        "label_column": "case_outcome"
    },
    "darklord1611/ecom_categories": {
        "text_column": "text",
        "label_column": "label"
    },
    "darklord1611/stackoverflow_question_ratings": {
        "text_column": "Body",
        "label_column": "Y"
    }
}
