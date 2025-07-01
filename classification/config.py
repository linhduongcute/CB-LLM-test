import concepts

example_name = {'SetFit/sst2': 'text', "fancyzhx/ag_news": 'text', "fancyzhx/yelp_polarity": 'text', "fancyzhx/dbpedia_14": 'content', "TimSchopf/medical_abstracts": 'medical_abstract', "darklord1611/legal_citations": 'case_text', "darklord1611/ecom_categories": 'text'} # Done
concepts_from_labels = {'SetFit/sst2': ["negative","positive"], "fancyzhx/yelp_polarity": ["negative","positive"], "fancyzhx/ag_news": ["World", "Sports", "Business", "Sci/Tech"], "fancyzhx/dbpedia_14": [
        "company",
        "educational institution",
        "artist",
        "athlete",
        "office holder",
        "mean of transportation",
        "building",
        "natural place",
        "village",
        "animal",
        "plant",
        "album",
        "film",
        "written work"
    ], "TimSchopf/medical_abstracts": ["neoplasms", "digestive_system_diseases", "nervous_system_diseases", "cardiovascular_diseases", "general_pathological_diseases"], "darklord1611/legal_citations": ["affirmed", "applied", "approved", "cited", "considered", "discussed", "distinguished", "followed", "referred to", "related"], "darklord1611/ecom_categories": ["Household", "Books", "Electronics", "Clothing & Accsessories"]} # Done

class_num = {'SetFit/sst2': 2, "fancyzhx/ag_news": 4, "fancyzhx/yelp_polarity": 2, "fancyzhx/dbpedia_14": 14, "TimSchopf/medical_abstracts": 5, "darklord1611/legal_citations": 10, "darklord1611/ecom_categories": 4} # Done

# Config for Roberta-Base baseline
finetune_epoch = {'SetFit/sst2': 3, "fancyzhx/ag_news": 2, "fancyzhx/yelp_polarity": 2, "fancyzhx/dbpedia_14": 2}
finetune_mlp_epoch = {'SetFit/sst2': 30, "fancyzhx/ag_news": 5, "fancyzhx/yelp_polarity": 3, "fancyzhx/dbpedia_14": 3}

# Config for CBM training
concept_set = {'SetFit/sst2': concepts.sst2, "fancyzhx/yelp_polarity": concepts.yelpp, "fancyzhx/ag_news": concepts.agnews, "fancyzhx/dbpedia_14": concepts.dbpedia, "TimSchopf/medical_abstracts": concepts.med_abs, "darklord1611/legal_citations": concepts.legal, "darklord1611/ecom_categories": concepts.ecom}


cbl_epochs = {'SetFit/sst2': 10, "fancyzhx/ag_news": 3, "fancyzhx/yelp_polarity": 2, "fancyzhx/dbpedia_14": 2, "TimSchopf/medical_abstracts": 10, "darklord1611/legal_citations": 10, "darklord1611/ecom_categories": 10}

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
