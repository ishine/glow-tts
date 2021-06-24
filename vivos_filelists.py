import os
import io

train_file = 'vivos/train/train.txt'
test_file = 'vivos/test/test.txt'
root = '../datasets'

def get_speakers_dict(file_path):
    speakers = set()
    file_path = os.path.join(root, file_path)
    with io.open(file_path) as f:
        for line in f.readlines():
            if len(line) < 2:
                continue
            speaker = os.path.basename(line.split('|')[0])
            speaker = speaker.split('_')[0]
            speakers.add(speaker)
    speakers = sorted(list(speakers))
    print(len(speakers))
    d = {}
    for i, name in enumerate(speakers):
        d[name] = str(i)
    return d


def create_filelists(root):
    train_path = os.path.join(root, train_file)
    test_path = os.path.join(root, test_file)

    train_d = get_speakers_dict(train_path)
    test_d = get_speakers_dict(test_path)

    with io.open(train_path, 'r', encoding='utf-8') as f:
        train_content = [l.split('|') for l in f if len(l) > 1]
    with io.open(test_path, 'r', encoding='utf-8') as f:
        test_content = [l.split('|') for l in f if len(l) > 1]
    
    with io.open('filelists/vivos_train.txt', 'w', encoding='utf-8') as f:
        for path, text in train_content:
            filename = os.path.basename(path)
            speaker = filename.split('_')[0]
            p1 = os.path.join(root, 'vivos/train/waves', speaker, filename)
            p2 = train_d[speaker]
            f.write(p1 + '|' + p2 + '|' + text) # text has '\n' already
    with io.open('filelists/vivos_test.txt', 'w', encoding='utf-8') as f:
        for path, text in test_content:
            filename = os.path.basename(path)
            speaker = filename.split('_')[0]
            p1 = os.path.join(root, 'vivos/test/waves', speaker, filename)
            p2 = test_d[speaker]
            f.write(p1 + '|' + p2 + '|' + text) # text has '\n' already



create_filelists(root)

