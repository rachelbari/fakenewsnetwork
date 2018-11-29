import os
from nltk.parse import stanford
from nltk.tree import Tree
os.environ[
    'STANFORD_PARSER'] = "/home/accts/sw852/final_project_copy/stanford-parser-full-2018-02-27/"
os.environ[
    'STANFORD_MODELS'] = "/home/accts/sw852/final_project_copy/stanford-parser-full-2018-02-27/"


def parse_data(directory, type):
    parser = stanford.StanfordParser(
        path_to_models_jar=
        "/home/accts/sw852/final_project_copy/stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar"
    )
    if type == "train":
        dest = open("{}parses/parsed_train.txt".format(os.fsdecode(directory)),
                    "w+")
    elif type == "dev":
        dest = open("{}parses/parsed_dev.txt".format(os.fsdecode(directory)),
                    "w+")
    else:
        dest = open("{}parses/parsed_test.txt".format(os.fsdecode(directory)),
                    "w+")

    filter_rep1 = open(
        "C:/Users/sarah/Dropbox/YaleSpring2018/LING380/final_project/rep_convote_train.txt"
    ).read()
    filter_dem1 = open(
        "C:/Users/sarah/Dropbox/YaleSpring2018/LING380/final_project/dem_convote_train.txt"
    ).read()
    filter_rep2 = open(
        "C:/Users/sarah/Dropbox/YaleSpring2018/LING380/final_project/rep_convote_test.txt"
    ).read()
    filter_dem2 = open(
        "C:/Users/sarah/Dropbox/YaleSpring2018/LING380/final_project/dem_convote_test.txt"
    ).read()

    for f in os.listdir(directory):
        filename = os.fsdecode(f)
        dir = os.fsdecode(directory)
        src = open("{}{}".format(dir, filename))
        for line in src:
            sentences = line.split('  ')
            for sent in sentences:
                if sent not in "\n \t  .,":
                    if sent in filter_rep1 or sent in filter_dem1 or sent in filter_rep2 or sent in filter_dem2:
                        parse = list(parser.raw_parse(sent))
                        tree = parse[0].pformat(nodesep=':')
                        ptree = ''.join(
                            [c if c != '\n' else ' ' for c in tree])
                        ptree = ''.join(
                            [c if c != '(' else ' ( ' for c in ptree])
                        ptree = ''.join(
                            [c if c != ')' else ' ) ' for c in ptree])
                        dest.write("{} x--x {}\n".format(filename, ptree))
        src.close()
        os.remove("{}{}".format(dir, filename))


def load_data(directory, type):
    corpus = []
    src = open("{}parses/parsed_{}.txt".format(directory, type))
    for line in src:
        entry = {}
        infos, tree = line.split('  ')
        entry['bill'] = infos[0]
        entry['speaker'] = infos[1]
        entry['party'] = infos[3][0]
        entry['mention'] = infos[3][1]
        entry['vote'] = infos[3][2]
        entry['tree'] = tree
        entry['utter'] = ' '.join(tree.leaves())
        corpus += [entry]
    return corpus


def load_all():
    data_dir = "C:/Users/sarah/Dropbox/YaleSpring2018/LING380/final_project/convote_v1.1/project_data/"

    training_data_dir = "{}training_set/".format(data_dir)
    training_directory = os.fsencode(training_data_dir)
    print("Loading training corpus.")
    training_corpus = load_data(training_directory, "train")
    print("Successfully loaded training set with {} examples.".format(
        len(training_corpus)))

    development_data_dir = "{}development_set/".format(data_dir)
    dev_dir = os.fsencode(development_data_dir)
    print("Loading development corpus.")
    dev_corpus = load_data(dev_dir, "dev")
    print("Successfully loaded development set with {} examples.".format(
        len(dev_corpus)))

    test_data_dir = "{}test_set/".format(data_dir)
    test_dir = os.fsencode(test_data_dir)
    print("Loading test corpus.")
    test_corpus = load_data(test_dir, "test")
    print("Successfully loaded test set with {} examples.".format(
        len(test_corpus)))

    return training_corpus, dev_corpus, test_corpus


def parse_all():
    data_dir = "C:/Users/sarah/Dropbox/YaleSpring2018/LING380/final_project/convote_v1.1/project_data/"

    training_data_dir = "{}training_set/".format(data_dir)
    training_directory = os.fsencode(training_data_dir)
    print("Parsing raw training sentences.")
    # parse_data(training_directory, "train")
    print("Finished parsing training set.")

    development_data_dir = "{}development_set/".format(data_dir)
    dev_dir = os.fsencode(development_data_dir)
    print("Parsing raw development sentences.")
    # parse_data(dev_dir, "dev")
    print("Finished parsing development set.")

    test_data_dir = "{}test_set/".format(data_dir)
    test_dir = os.fsencode(test_data_dir)
    print("Parsing raw test sentences.")
    parse_data(test_dir, "test")
    print("Finished parsing test set.")


def add_parses(directory):
    dest = open("/home/accts/sw852/final_project_copy/parsed_extreme.txt.",
                "w+")
    count = 0
    for f in os.listdir(os.fsencode(directory)):
        filename = os.fsdecode(f)
        src = open("{}{}".format(directory, filename))
        for tree in src:
            if "yield" not in tree and ("support" in tree or "oppose" in tree):
                # print(tree)
                ptree = ''.join([
                    c if c != '[' and c != ']' and c != '\n' and c != '\'' else
                    ' ' for c in tree
                ])
                pptree = ""
                for i in range(len(ptree) - 2):
                    if ptree[i + 2] == ',':
                        pptree += ptree[i] + ":"
                        i += 2
                    else:
                        pptree += ptree[i]
                pptree = ''.join([c if c != ',' else '' for c in pptree])
                pptree = ''.join(pptree.split('Tree'))
                # print(pptree)
                dest.write("{} x--x {}\n".format(filename, pptree))
                count += 1
        src.close()
    print("Added {} trees".format(count))
    dest.close()


if __name__ == '__main__':

    add_parses("/home/accts/sw852/final_project_copy/parses/")
    # parse_all()
    pass
