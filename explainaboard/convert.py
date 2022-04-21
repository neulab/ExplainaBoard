import csv


def convert_cls():
    origin = "data/system_outputs/ted_multi/ted_multi_slk_eng.nmt"
    dataset = "data/system_outputs/ted_multi/ted_multi_slk_eng_dataset.tsv"
    output = "data/system_outputs/ted_multi/ted_multi_slk_eng_output.nmt"
    with open(origin, "r") as f1:
        data = []
        out = []
        for d in csv.reader(f1, delimiter="\t"):
            data.append(d[:2])
            out.append(d[2])

        with open(dataset, 'w') as f2:
            writer = csv.writer(f2, delimiter="\t")
            writer.writerows(data)
        with open(output, "w") as f3:
            f3.writelines([line + "\n" for line in out])


def convert_text_pair():
    origin = "data/system_outputs/snli/snli.bert"
    dataset = "data/system_outputs/snli/snli_dataset.tsv"
    output = "data/system_outputs/snli/snli_output_bert.txt"
    with open(origin, "r") as f1:
        data = []
        out = []
        for d in csv.reader(f1, delimiter="\t"):
            if len(d) == 2:
                print(d)
            data.append(d[:3])
            out.append(d[3])
        with open(dataset, 'w') as f2:
            writer = csv.writer(f2, delimiter="\t")
            writer.writerows(data)
        with open(output, "w") as f3:
            f3.writelines([line + "\n" for line in out])


if __name__ == "__main__":
    convert_cls()
