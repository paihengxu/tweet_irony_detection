Work flow includes -
1. Environment setup and run needed servers
2. Feature generation and saving
3. Running specific combinations from report

============================================================================

1. All needed packages are in cs666.yml and also requirements.txt
    1.1 conda env create -f cs666.yml
    1.2 Download stanfordNLP package from https://stanfordnlp.github.io/CoreNLP/, and unzip the stanford-corenlp-full-2018-10-05 into root path
        Activate stanfordNLP server with below commands or use run.sh
        cd stanford-corenlp-full-2018-10-05;
        java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 5000 -timeout 10000

2. Feature generation and saving
For ease of experimenting with features. we generate the features and save them to compressed jsons individually.
They are used later to build combination features.

From top level project directory -

    2.1. Genetrate baseline features
            python -m baseline.features
            This should create 10 features.json files for each of A and B's training and testing data.
            Move them to obtained_features/baselines.
            NOTE: Since you are pulling the project from git they might already be there as they are small files and we are able to push them
    2.2. Generate behaviour features
            python -m behavior_model.features
            This should create 5 features.json files for each of A and B's training and testing data.
            Move them to obtained_features/behavior_model.
            NOTE: Since you are pulling the project from git they might already be there as they are small files and we are able to push them

    2.3. Generate word embedding features
            BERT: python -m word_representations.bert
            ELMo: python -m word_representations.elmo
            Skip-gram:
            CBOW:

            These are large files and couldn't be pushed to git. But inlucded in zipped code.
            These json.gz files should be in project top level directory

    2.4 Generating BERT Fine Tuned


3. Running classifiers and print metrics

From top level project directory -

    3.1 Baseline + Logistic regression

    3.2 Baseline  + Behavior + Logistic regression

    3.3 Baseline + Behavior + PCA + logistic Regression

    3.4 Word Embeddings + Logistic Regression
        pyhton -m hybrid_combinations.combo1_lr

    3.5 Word Embeddings + MLP
        pyhton -m hybrid_combinations.combo1

    3.5 BERT Fine Tuned + LR

    3.6 All features + LR

    3.7 All features + PCA + LR
