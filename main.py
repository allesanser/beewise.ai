import sys
import pandas as pd
from navec import Navec
import torch
import torch.nn as nn
from yargy import Parser, rule, or_
from yargy.predicates import gram, dictionary


path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)
cos = nn.CosineSimilarity()
info_dict = {}


RULE_FOR_NAME = or_(rule(dictionary({'меня', 'это', 'я'}),
                 gram('Name'),),
          rule(dictionary({'зовут'}),
               gram('Name')))

PARSER_FOR_NAME = Parser(RULE_FOR_NAME)


def simular_greating(text):
      global info_dict
      if text.dlg_id not in info_dict:
            info_dict[text.dlg_id] = [[],[],[],[],[],0]

      
      for word in text.text.split():
          # print(word))
          if word in navec:
              if cos(torch.tensor([navec[word]]), torch.tensor([navec['привет']])).item() > 0.63:
                  info_dict[text.dlg_id][0].append(text.text)
                  break
              elif cos(torch.tensor([navec[word]]), torch.tensor([navec['здравствуйте']])).item() > 0.52:
                  info_dict[text.dlg_id][0].append(text.text)
                  break

def simular_name(text):
      global info_dict
      name = 0

      for word in text.text.split():
          # print(word))
          wordd = ''
          if word in navec:
              if cos(torch.tensor([navec[word]]), torch.tensor([navec['дмитрий']])).item() > 0.5:
                  for match in PARSER_FOR_NAME.findall(text.text):
                        wordd = [x.value for x in match.tokens][-1]
                  if len(wordd) > 1:           
                      info_dict[text.dlg_id][1].append(text.text)
                      info_dict[text.dlg_id][2].append(wordd)
              elif cos(torch.tensor([navec[word]]), torch.tensor([navec['анастасия']])).item() > 0.5:
                  for match in PARSER_FOR_NAME.findall(text.text):
                        wordd = [x.value for x in match.tokens][-1]
                  if len(wordd) > 1:           
                      info_dict[text.dlg_id][1].append(text.text)
                      info_dict[text.dlg_id][2].append(wordd)


def company_name(text):
      global info_dict

      sentence = text.text.split()
      company_name = ''
      for k in range(len(sentence)):
          # print(word))
          if sentence[k] in navec:
              if cos(torch.tensor([navec[sentence[k]]]), torch.tensor([navec['название'] + navec['компании']])).item() > 0.5:
                  perem = 1
                  if len(sentence) - 3 > k:
                        for _ in range(1,4):
                            if sentence[k + _] in navec:
                                if cos(torch.tensor([navec[sentence[k + _]]]), torch.tensor([navec['компания']])).item() > 0.4:
                                      company_name = company_name + ' ' + sentence[k+_]
                            else:
                                company_name = company_name + ' ' + sentence[k+_]
                  elif len(sentence) - 2 > k:
                      for _ in range(1,3):
                            if sentence[k + _] in navec:
                                if cos(torch.tensor([navec[sentence[k + _]]]), torch.tensor([navec['компания']])).item() > 0.4:
                                      company_name = company_name + ' ' + sentence[k+_]
                            else:
                                company_name = company_name + ' ' + sentence[k + _]
                  if len(company_name)>1:
                      info_dict[text.dlg_id][3].append(company_name[1:])
                  break


def simular_goodbye(text):
      global info_dict
      perem = 0
      for word in text.text.split():
          # print(word))
          if word in navec:
              if cos(torch.tensor([navec[word]]), torch.tensor([navec['до'] + navec['свидания']])).item() > 0.63 and word != 'до':
                  # print(word)
                  if len(info_dict[text.dlg_id][0]) > 0:
                        info_dict[text.dlg_id][5] = 1
                  info_dict[text.dlg_id][4].append(text.text)
                  perem = 1
                  break
              elif cos(torch.tensor([navec[word]]), torch.tensor([navec['доброго'] + navec['вечера']])).item() > 0.55:
                  if len(info_dict[text.dlg_id][0]) > 0:
                        info_dict[text.dlg_id][5] = 1
                  info_dict[text.dlg_id][4].append(text.text)
                  # print(word)
                  perem = 1
                  break


if __name__ == "__main__":
    
        path_to_data = 'test_data.csv'
        
        if len(sys.argv) < 2:
                print("working with default dataset")
        else:
                path_to_data =  sys.argv[1]
                
                
        data = pd.read_csv(path_to_data) 
        print("Данные считаны") 
        
        data[data['role'] == 'manager'].apply(lambda x: simular_greating(x), axis=1)
        data[data['role'] == 'manager'].apply(lambda x: simular_name(x), axis=1)
        data[data['role'] == 'manager'].apply(lambda x: company_name(x), axis=1)
        data[data['role'] == 'manager'].apply(lambda x: simular_goodbye(x),axis=1)
        
        for k in info_dict:
            print(k, ' - dlg_id')
            for line in info_dict[k]:
                    print(line)
            print()
        