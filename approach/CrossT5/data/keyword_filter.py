import os
def main():
    data_path = "F:\\Asuna\\Desktop\\datasets\\study_datasets\\5_cross_trans_datasets\\file_test_0_Dataset"

    testa = os.path.join(data_path, "Testing/assertLines.txt")
    traina = os.path.join(data_path, "Training/assertLines.txt")
    token = list()
    with open(testa, 'r', encoding="utf-8") as f:
        data = f.readlines()
        for i in data:
            token.extend([j.strip() for j in i.split()])
    with open(traina, 'r', encoding="utf-8") as f:
        data = f.readlines()
        for i in data:
            token.extend([j.strip() for j in i.split()])
    token = set(token)
    # print(token)
    keywords = {'null', 'native', 'this', '{', '==', 'if', 'protected', '!=', 'transient', '=', 'static', ']', 'import', '/', '++',
     '}', 'goto', 'enum', '>>>', 'package', '&=', 'byte', 'false', '>>', '|', '>', 'true', '+=', ')', '^', '%=', 'char',
     'synchronized', 'extends', '!', '~', '[', 'transitive', '/=', 'strictfp', 'while', 'interface', '<', '@', 'int',
     '<=', '--', '&&', 'final', '%', 'float', 'implements', 'double', '>>=', '.', '<<', 'try', 'new', 'return',
     'private', '$', '*', 'throw', 'long', 'throws', 'public', 'instanceof', '-', 'continue', 'short', '-=', '+', ';',
     '<<==', 'abstract', 'else', 'volatile', '&', 'do', 'for', 'case', 'boolean', 'default', 'void', 'break', 'assert',
     '>>>=', 'finally', '>=', 'class', '||', '^=', 'switch', 'super', '|=', '(', 'const', ':', 'catch'}
    keywords = set(keywords)
    print(keywords - token)
    print(keywords)

if __name__ == "__main__":
    main()