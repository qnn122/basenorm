import torch
from os import listdir, write
from os.path import isfile, join, splitext
import copy
from pronto import Ontology
import copy
import os
from src.utils import lowercaser_mentions, lowercaser_ref

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.mention = X
        self.label = y

    def __getitem__(self, idx):
        mention = self.mention[idx]
        label = self.label[idx]
        sample = (mention, label)
        return sample

    def __len__(self):
        return len(self.mention)
    

################################################
# LOADING DATASET:
################################################
def load_bb4_data(bb4_folder_path):
	obo_file_path = os.path.join(bb4_folder_path, "OntoBiotope_BioNLP-OST-2019.obo")
	train_folder_path = os.path.join(bb4_folder_path, "BioNLP-OST-2019_BB-norm_train")
	dev_folder_path = os.path.join(bb4_folder_path, "BioNLP-OST-2019_BB-norm_dev")
	test_folder_path = os.path.join(bb4_folder_path, "BioNLP-OST-2019_BB-norm_test")

	print("loading OntoBiotope...")
	dd_obt = loader_ontobiotope(obo_file_path)
	print("loaded. (Nb of concepts in OBT =", len(dd_obt.keys()), ")")

	print("\nExtracting Bacterial Habitat hierarchy:")
	dd_habObt = select_subpart_hierarchy(dd_obt, 'OBT:000001')
	print("Done. (Nb of concepts in this subpart of OBT =", len(dd_habObt.keys()), ")")

	print("\nLoading BB4 corpora...")
	ddd_dataAll = loader_one_bb4_fold([train_folder_path, dev_folder_path, test_folder_path])
	dd_habAll = extract_data(ddd_dataAll, l_type=["Habitat"])
	print("loaded.(Nb of mentions in whole corpus =", len(dd_habAll.keys()), ")")

	ddd_dataTrain = loader_one_bb4_fold([train_folder_path])
	dd_habTrain = extract_data(ddd_dataTrain, l_type=["Habitat"])  # ["Habitat", "Phenotype", "Microorganism"]
	print("loaded.(Nb of mentions in train =", len(dd_habTrain.keys()), ")")

	ddd_dataDev = loader_one_bb4_fold([dev_folder_path])
	dd_habDev = extract_data(ddd_dataDev, l_type=["Habitat"])
	print("loaded.(Nb of mentions in dev =", len(dd_habDev.keys()), ")")

	ddd_dataTrainDev = loader_one_bb4_fold([train_folder_path, dev_folder_path])
	dd_habTrainDev = extract_data(ddd_dataTrainDev, l_type=["Habitat"])
	print("loaded.(Nb of mentions in train+dev =", len(dd_habTrainDev.keys()), ")")

	ddd_dataTest = loader_one_bb4_fold([test_folder_path])
	dd_habTest = extract_data(ddd_dataTest, l_type=["Habitat"])
	print("loaded.(Nb of mentions in test =", len(dd_habTest.keys()), ")")

	################################################
	print("\n\nPREPROCESSINGS:\n")
	################################################

	print("Mentions lowercasing...")
	dd_BB4habTrain_lowercased = lowercaser_mentions(dd_habTrain)
	dd_BB4habDev_lowercased = lowercaser_mentions(dd_habDev)
	print("Done.\n")

	print("Lowercase references...")
	dd_habObt_lowercased = lowercaser_ref(dd_habObt)

	dd_ref = dd_habObt_lowercased
	dd_train = dd_BB4habTrain_lowercased
	dd_test = dd_BB4habDev_lowercased
	print("Done.")
	return dd_ref, dd_train, dd_test, dd_habDev


###################################################
# BB4 normalization dataset loader:
###################################################
def loader_one_bb4_fold(l_repPath):
    """
    Description: Load BB4 data from files.
    WARNING: OK only if A1 file is read before its A2 file (normally the case).
    :param l_repPath: A list of directory path containing set of A1 (and possibly A2) files.
    :return:
    """
    ddd_data = dict()

    i = 0
    for repPath in l_repPath:

        for fileName in listdir(repPath):
            filePath = join(repPath, fileName)

            if isfile(filePath):

                fileNameWithoutExt, ext = splitext(fileName)
                
                if ext == ".a1":

                    with open(filePath, encoding="utf8") as file:
                    
                        if fileNameWithoutExt not in ddd_data.keys():

                            ddd_data[fileNameWithoutExt] = dict()
                            for line in file:

                                l_line = line.split('\t')

                                if l_line[1].split(' ')[0] == "Title" or l_line[1].split(' ')[0] == "Paragraph":
                                    pass
                                else:
                                    exampleId = "bb4_" + "{number:06}".format(number=i)

                                    ddd_data[fileNameWithoutExt][exampleId] = dict()

                                    ddd_data[fileNameWithoutExt][exampleId]["T"] = l_line[0]
                                    ddd_data[fileNameWithoutExt][exampleId]["type"] = l_line[1].split(' ')[0]
                                    ddd_data[fileNameWithoutExt][exampleId]["mention"] = l_line[2].rstrip()

                                    if "cui" not in ddd_data[fileNameWithoutExt][exampleId].keys():
                                        ddd_data[fileNameWithoutExt][exampleId]["cui"] = list()
                                    i += 1

        for fileName in listdir(repPath):
            filePath = join(repPath, fileName)

            if isfile(filePath):

                fileNameWithoutExt, ext = splitext(fileName)
                
                if ext == ".a2":

                    with open(filePath, encoding="utf8") as file:

                        if fileNameWithoutExt in ddd_data.keys():

                            for line in file:
                                l_line = line.split('\t')

                                l_info = l_line[1].split(' ')
                                Tvalue = l_info[1].split(':')[1]

                                for id in ddd_data[fileNameWithoutExt].keys():
                                    if ddd_data[fileNameWithoutExt][id]["T"] == Tvalue:
                                        if ddd_data[fileNameWithoutExt][id]["type"] == "Habitat" or \
                                                ddd_data[fileNameWithoutExt][id]["type"] == "Phenotype":
                                            cui = "OBT:" + l_info[2].split('Referent:')[1].rstrip().replace('OBT:', '')
                                            ddd_data[fileNameWithoutExt][id]["cui"].append(cui)
                                        elif ddd_data[fileNameWithoutExt][id]["type"] == "Microorganism":
                                            cui = l_info[2].split('Referent:')[1].rstrip()
                                            ddd_data[fileNameWithoutExt][id]["cui"] = [cui]  # No multi-normalization for microorganisms
    return ddd_data


def extract_data(ddd_data, l_type=[]):
    """

    :param ddd_data:
    :param l_type:
    :return:
    """
    dd_data = dict()

    for fileName in ddd_data.keys():
        for id in ddd_data[fileName].keys():
            if ddd_data[fileName][id]["type"] in l_type:
                dd_data[id] = copy.deepcopy(ddd_data[fileName][id])
    return dd_data


###################################################
# Ontological tools:
###################################################

def loader_ontobiotope(filePath):
    """
    Description: A loader of OBO ontology based on Pronto lib.
    (maybe useless...)
    :param filePath: Path to the OBO file.
    :return: an annotation ontology in a dict (format: concept ID (string): {'label': preferred tag,
    'tags': list of other tags, 'parents': list of parents concepts}
    """
    dd_obt = dict()
    onto = Ontology(filePath).terms()
    for o_concept in onto:
        dd_obt[o_concept.id] = dict()
        dd_obt[o_concept.id]["label"] = o_concept.name
        dd_obt[o_concept.id]["tags"] = list()

        for o_tag in o_concept.synonyms:
            dd_obt[o_concept.id]["tags"].append(o_tag.description)

        dd_obt[o_concept.id]["parents"] = list()
        parents = o_concept.superclasses(with_self=False, distance=1).to_set()
        for o_parent in parents:
            dd_obt[o_concept.id]["parents"].append(o_parent.id)

    return dd_obt


def is_desc(dd_ref, cui, cuiParent):
    """
    Description: A function to get if a concept is a descendant of another concept.
    Here, only used to select a clean subpart of an existing ontology (see select_subpart_hierarchy method).
    """
    result = False
    if "parents" in dd_ref[cui].keys():
        if len(dd_ref[cui]["parents"]) > 0:
            if cuiParent in dd_ref[cui]["parents"]:  # Not working if infinite is_a loop (normally never the case!)
                result = True
            else:
                for parentCui in dd_ref[cui]["parents"]:
                    result = is_desc(dd_ref, parentCui, cuiParent)
                    if result:
                        break
    return result


def select_subpart_hierarchy(dd_ref, newRootCui):
    """
    Description: By picking a single concept in an ontology, create a new sub ontology with this concept as root.
    Here, only used to select the habitat subpart of the Ontobiotope ontology.
    """
    dd_subpart = dict()
    dd_subpart[newRootCui] = copy.deepcopy(dd_ref[newRootCui])
    dd_subpart[newRootCui]["parents"] = []

    for cui in dd_ref.keys():
        if is_desc(dd_ref, cui, newRootCui):
            dd_subpart[cui] = copy.deepcopy(dd_ref[cui])

    # Clear concept-parents which are not in the descendants of the new root:
    for cui in dd_subpart.keys():
        dd_subpart[cui]["parents"] = list()
        for parentCui in dd_ref[cui]["parents"]:
            if is_desc(dd_ref, parentCui, newRootCui) or parentCui == newRootCui:
                dd_subpart[cui]["parents"].append(parentCui)

    return dd_subpart

