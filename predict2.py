import os
import json
from preprocessing import preprocess_sentence
from utils import *
from config import *

def predict_sentence(entities_model, intents_model, tokens):
    with torch.no_grad():    
        # Get predictions from the model
        output_entities = entities_model(tokens)
        output_intents = intents_model(tokens)

        # Predicted entities and intents
        pred_entities = torch.argmax(output_entities, dim=-1)
        pred_intents = torch.argmax(output_intents, dim=-1)

        return pred_entities, pred_intents
    

########################################################################################################################
def get_pizza_details(order, token, entity_type, toppings, styles):
    if (entity_type=='BNUMBER'):
        order['NUMBER'] = token
    elif (entity_type=='INUMBER'):
        if(order['NUMBER'] is not None):
            order['NUMBER'] += " " + token
        else:
            order['NUMBER'] = token
    elif (entity_type=='BSIZE'):
        order['SIZE'] = token
    elif (entity_type=='ISIZE'):
        if(order['SIZE'] is not None):
            order['SIZE'] += " " + token
        else:
            order['SIZE'] = token
    elif (entity_type=='BSTYLE'):
        styles['TYPE'] = token
    elif (entity_type=='ISTYLE'):
        if(styles['TYPE'] is not None):
            styles['TYPE'] += " " + token
        else:
            styles['TYPE'] = token
    elif (entity_type=='STYLE_NOT'): 
        styles['NOT'] += True
    elif (entity_type=='BQUANTITY'):
        toppings['Quantity'] = token
    elif (entity_type=='IQUANTITY'): 
        if(toppings['Quantity'] is not None):
            toppings['Quantity'] += " " + token
        else:
            toppings['Quantity'] = token
    elif (entity_type=='BTOPPING'):
        toppings['Topping'] = token
    elif (entity_type=='ITOPPING'): 
        if(toppings['Topping'] is not None):
            toppings['Topping'] += " " + token
        else:
            toppings['Topping'] = token
    elif (entity_type=='TOPPING_NOT'): 
        toppings['NOT'] += True


    return order, toppings, styles

def get_drink_details(order, token, entity_type):
    if (entity_type=='BNUMBER'):
        order['NUMBER'] = token
    elif (entity_type=='INUMBER'):
        if(order['NUMBER'] is not None):
            order['NUMBER'] += " " + token
        else:
            order['NUMBER'] = token
    elif (entity_type=='BSIZE'):
        order['SIZE'] = token
    elif (entity_type=='ISIZE'):
        if(order['SIZE'] is not None):
            order['SIZE'] += " " + token
        else:
            order['SIZE'] = token
    elif (entity_type=='BDRINKTYPE'):
        order['DRINKTYPE'] = token
    elif (entity_type=='IDRINKTYPE'):
        if(order['DRINKTYPE'] is not None):
            order['DRINKTYPE'] += " " + token
        else:
            order['DRINKTYPE'] = token
    elif (entity_type=='BVOLUME'):
        order['VOLUME'] = token
    elif (entity_type=='IVOLUME'):
        if(order['VOLUME'] is not None):
            order['VOLUME'] += " " + token
        else:
            order['VOLUME'] = token
    elif (entity_type=='BCONTAINERTYPE'):
        order['CONTAINERTYPE'] = token
    elif (entity_type=='ICONTAINERTYPE'):
        if(order['CONTAINERTYPE'] is not None):
            order['CONTAINERTYPE'] += " " + token
        else:
            order['CONTAINERTYPE'] = token

    return order
    

def pizza_check(entity, prev_intent):
    entity_check = (entity=="BSYTLE" or entity=="ISTYLE" or entity=="STYLE_NOT" or entity=="BQUANTITY" or entity=="IQUANTITY" or entity=="BTOPPING" or entity=="ITOPPING" or entity=="TOPPING_NOT")
    intent_check = (prev_intent=="BPIZZAORDER" or prev_intent=="IPIZZAORDER" or prev_intent=="BCOMPLEX_TOPPING" or prev_intent=="ICOMPLEX_TOPPING")
    return entity_check 

def drink_check(entity):
    entity_check = (entity=="BVOLUME" or entity=="IVOLUME" or entity=="BCONTAINERTYPE" or entity=="ICONTAINERTYPE" or entity=="BDRINKTYPE" or entity=="IDRINKTYPE")
    return entity_check 
    


def get_orders(tokens, predicted_entities, predicted_intents, reverse_entities_labels_map, reverse_intents_labels_map):
    all_pizza_orders = []
    all_drink_orders = []
    pizza_order = {"NUMBER": None, "SIZE": None, "STYLE": None, "AllTopping": None}
    drink_order = {"NUMBER": None, "SIZE": None, "DRINKTYPE": None, "VOLUME":None, "CONTAINERTYPE": None}
    pizza_topping = {"NOT": False, "Quantity": None, "Topping": None}
    pizza_style = {"NOT": False, "TYPE": None}
    all_pizza_toppings = []
    all_pizza_styles = []
    pizza_begin_flag = drink_begin_flag = True

    prev_entity = curr_entity = 'NONE'
    prev_intent = curr_intent = 'NONE'
    for token, entity, intent in zip(tokens, predicted_entities, predicted_intents):
        entity_type = reverse_entities_labels_map[int(entity)]
        intent_type = reverse_intents_labels_map[int(intent)]
        #print(token,entity_type,intent_type)

        prev_entity = curr_entity
        curr_entity = entity_type
        prev_intent = curr_intent
        curr_intent = intent_type
        if (intent_type=='BPIZZAORDER') or (intent_type=='IPIZZAORDER') or (intent_type=='BCOMPLEX_TOPPING') or (intent_type=='ICOMPLEX_TOPPING') or pizza_check(entity_type,prev_intent): # Pizza Order
            #print(prev_intent,"->",curr_intent)
            if ((prev_intent=='NONE') and (curr_intent=='IPIZZAORDER')) or intent_type=='BPIZZAORDER' or ((prev_intent=='NONE') and (curr_intent=='BCOMPLEX_TOPPING')):  # Beginning of a new pizza_order
                if pizza_begin_flag:
                    pizza_order = {"NUMBER": None, "SIZE": None, "STYLE": None, "AllTopping": None}
                    pizza_topping = {"NOT": False, "Quantity": None, "Topping": None}
                    pizza_style = {"NOT": False, "TYPE": None}
                    all_pizza_toppings = []
                    all_pizza_styles = []
                    pizza_begin_flag = False
                else:
                    pizza_order['STYLE'] = all_pizza_styles
                    pizza_order['AllTopping'] = all_pizza_toppings
                    all_pizza_orders.append(pizza_order)
                    pizza_order = {"NUMBER": None, "SIZE": None, "STYLE": None, "AllTopping": None}
                    all_pizza_toppings = []
                    all_pizza_styles = []
                    pizza_topping = {"NOT": False, "Quantity": None, "Topping": None}
                    pizza_style = {"NOT": False, "TYPE": None}

            pizza_order, pizza_topping, pizza_style =  get_pizza_details(pizza_order, token, entity_type, pizza_topping, pizza_style)
            if (prev_entity == 'BTOPPING' and curr_entity != 'ITOPPING') or (prev_entity == 'ITOPPING' and curr_entity != 'ITOPPING'):
                all_pizza_toppings.append(pizza_topping)
                pizza_topping = {"NOT": False, "Quantity": None, "Topping": None}
            if (prev_entity == 'BSTYLE' and curr_entity != 'ISTYLE') or (prev_entity == 'ISTYLE' and curr_entity != 'ISTYLE'):
                all_pizza_styles.append(pizza_style)
                pizza_style = {"NOT": False, "TYPE": None}

        elif (intent_type=='BDRINKORDER') or (intent_type=='IDRINKORDER'): # Pizza Order
            if ((prev_intent=='NONE') and (curr_intent=='IDRINKORDER')) or intent_type=='BDRINKORDER' or drink_check(entity_type):  # Beginning of a new drink_order
                if drink_begin_flag:
                    drink_order = {"NUMBER": None, "SIZE": None, "DRINKTYPE": None, "VOLUME":None, "CONTAINERTYPE": None}
                    drink_begin_flag = False
                else:
                    all_drink_orders.append(drink_order)
                    drink_order = {"NUMBER": None, "SIZE": None, "DRINKTYPE": None, "VOLUME":None, "CONTAINERTYPE": None}

            drink_order =  get_drink_details(drink_order, token, entity_type)

        #print(pizza_order, pizza_topping)

    if not pizza_begin_flag:
        if pizza_topping['Topping'] is not None:
            all_pizza_toppings.append(pizza_topping)
        pizza_order['AllTopping'] = all_pizza_toppings

        if pizza_style['TYPE'] is not None:
            all_pizza_styles.append(pizza_style)
        pizza_order['STYLE'] = all_pizza_styles

        all_pizza_orders.append(pizza_order)

    if not drink_begin_flag:
        all_drink_orders.append(drink_order)

    return all_pizza_orders, all_drink_orders


########################################################################################################################
def predict_JSON(device, model_name, data_size):
    entities_model = torch.load(os.path.join('./Models/Saved_Models', model_name+"_entities_"+str(data_size)+".pth"))
    intents_model = torch.load(os.path.join('./Models/Saved_Models', model_name+"_intents_"+str(data_size)+".pth"))

    # Load the dictionary from the file
    with open(os.path.join('./Dataset',"word_to_int_"+str(data_size)+".json"), "r") as file:
        word_to_int = json.load(file)

    with open("./Dataset/Input_Output/input.txt", "r") as file:
        sentence = file.read().strip()
     
    tokens, tokens_tokenized = preprocess_sentence(sentence, word_to_int)
    tokens_tokenized = torch.tensor(tokens_tokenized)
    tokens_tokenized = tokens_tokenized.to(device)

    pred_entities, pred_intents = predict_sentence(entities_model, intents_model, tokens_tokenized)
    #print(tokens, pred_entities)

    reverse_entities_labels_map = {v: k for k, v in entities_label_map.items()}
    reverse_intents_labels_map = {v: k for k, v in intents_label_map.items()}
    pizza_orders, drink_orders = get_orders(tokens, pred_entities, pred_intents,reverse_entities_labels_map,reverse_intents_labels_map)
    print(pizza_orders,drink_orders)
    order = {'PIZZAORDER':pizza_orders,'DRINKORDER':drink_orders}
    final_output = {"ORDER": order}
    with open("./Dataset/Input_Output/output.json", "w") as file:
        json.dump(final_output, file, indent=4)

def predict(device, model_name, data_size, sentence):
    entities_model = torch.load(os.path.join('./Models/Saved_Models', model_name+"_entities_"+str(data_size)+".pth"))
    intents_model = torch.load(os.path.join('./Models/Saved_Models', model_name+"_intents_"+str(data_size)+".pth"))

    # Load the dictionary from the file
    with open(os.path.join('./Dataset',"word_to_int_"+str(data_size)+".json"), "r") as file:
        word_to_int = json.load(file)

    tokens, tokens_tokenized = preprocess_sentence(sentence, word_to_int)
    tokens_tokenized = torch.tensor(tokens_tokenized)
    tokens_tokenized = tokens_tokenized.to(device)

    pred_entities, pred_intents = predict_sentence(entities_model, intents_model, tokens_tokenized)
    #print(tokens, pred_entities)

    reverse_entities_labels_map = {v: k for k, v in entities_label_map.items()}
    reverse_intents_labels_map = {v: k for k, v in intents_label_map.items()}
    pizza_orders, drink_orders = get_orders(tokens, pred_entities, pred_intents,reverse_entities_labels_map,reverse_intents_labels_map)
    order = {'PIZZAORDER':pizza_orders,'DRINKORDER':drink_orders}
    final_output = {"ORDER": order}
    
    return final_output