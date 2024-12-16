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
def get_pizza_details(order, token, entity_type, toppings):
    if (entity_type=='BNUMBER'):
        order['NUMBER'] = token
    elif (entity_type=='INUMBER'):
        order['NUMBER'] += " " + token
    elif (entity_type=='BSIZE'):
        order['SIZE'] = token
    elif (entity_type=='ISIZE'):
        order['SIZE'] += " " + token
    elif (entity_type=='BSTYLE'):
        order['STYLE'] = token
    elif (entity_type=='ISTYLE'):
        order['STYLE'] += " " + token
    elif (entity_type=='BQUANTITY'):
        toppings['Quantity'] = token
    elif (entity_type=='IQUANTITY'): 
        toppings['Quantity'] += " " + token
    elif (entity_type=='BTOPPING'):
        toppings['Topping'] = token
    elif (entity_type=='ITOPPING'): 
        toppings['Topping'] += " " + token
    elif (entity_type=='NOT'): 
        toppings['NOT'] += True

    return order, toppings

def get_drink_details(order, token, entity_type):
    if (entity_type=='BNUMBER'):
        order['NUMBER'] = token
    elif (entity_type=='INUMBER'):
        order['NUMBER'] += " " + token
    elif (entity_type=='BSIZE'):
        order['SIZE'] = token
    elif (entity_type=='ISIZE'):
        order['SIZE'] += " " + token
    elif (entity_type=='BSTYLE'):
        order['STYLE'] = token
    elif (entity_type=='ISTYLE'):
        order['STYLE'] += " " + token
    elif (entity_type=='BDRINKTYPE'):
        order['DRINKTYPE'] = token
    elif (entity_type=='IDRINKTYPE'):
        order['DRINKTYPE'] += " " + token
    elif (entity_type=='BCONTAINERTYPE'):
        order['CONTAINERTYPE'] = token
    elif (entity_type=='ICONTAINERTYPE'):
        order['CONTAINERTYPE'] += " " + token

    return order
    
def get_orders(tokens, predicted_entities, predicted_intents, reverse_entities_labels_map, reverse_intents_labels_map):
    all_pizza_orders = []
    all_drink_orders = []
    pizza_order = {"NUMBER": None, "SIZE": None, "STYLE": None, "AllTopping": None}
    drink_order = {"NUMBER": None, "SIZE": None, "DRINKTYPE": None, "CONTAINERTYPE": None}
    pizza_topping = {"NOT": False, "Quantity": None, "Topping": None}
    pizza_begin_flag = drink_begin_flag = True

    prev_entity = curr_entity = None
    for token, entity, intent in zip(tokens, predicted_entities, predicted_intents):
        entity_type = reverse_entities_labels_map[int(entity)]
        intent_type = reverse_intents_labels_map[int(intent)]
        print(token,entity_type,intent_type)

        prev_entity = curr_entity
        curr_entity = entity_type
        if (intent_type=='BPIZZAORDER') or (intent_type=='IPIZZAORDER') or (intent_type=='BCOMPLEX_TOPPING') or (intent_type=='ICOMPLEX_TOPPING'): # Pizza Order
            if intent_type=='BPIZZAORDER':  # Beginning of a new pizza_order
                if pizza_begin_flag:
                    pizza_order = {"NUMBER": None, "SIZE": None, "STYLE": None, "AllTopping": None}
                    pizza_topping = {"NOT": False, "Quantity": None, "Topping": None}
                    all_pizza_toppings = []
                    pizza_begin_flag = False
                else:
                    pizza_order['AllTopping'] = all_pizza_toppings
                    all_pizza_orders.append(pizza_order)
                    pizza_order = {"NUMBER": None, "SIZE": None, "STYLE": None, "AllTopping": None}
                    all_pizza_toppings = []
                    pizza_topping = {"NOT": False, "Quantity": None, "Topping": None}

            pizza_order, pizza_topping =  get_pizza_details(pizza_order, token, entity_type, pizza_topping)
            if (prev_entity == 'BTOPPING' and curr_entity != 'ITOPPING') or (prev_entity == 'ITOPPING' and curr_entity != 'ITOPPING'):
                all_pizza_toppings.append(pizza_topping)
                pizza_topping = {"NOT": False, "Quantity": None, "Topping": None}

        elif (intent_type=='BDRINKORDER') or (intent_type=='IDRINKORDER'): # Pizza Order
            if intent_type.startswith("B"):  # Beginning of a new pizza_order
                if drink_begin_flag:
                    drink_order = {"NUMBER": None, "SIZE": None, "DRINKTYPE": None, "CONTAINERTYPE": None}
                    drink_begin_flag = False
                else:
                    all_drink_orders.append(drink_order)
                    drink_order = {"NUMBER": None, "SIZE": None, "DRINKTYPE": None, "CONTAINERTYPE": None}

            drink_order =  get_drink_details(pizza_order, token, entity_type)

        print(pizza_order, pizza_topping)

    if not pizza_begin_flag:
        if pizza_topping['Topping'] is not None:
            all_pizza_toppings.append(pizza_topping)
        pizza_order['AllTopping'] = all_pizza_toppings
        all_pizza_orders.append(pizza_order)

    if not drink_begin_flag:
        all_drink_orders.append(drink_order)

    return all_pizza_orders, all_drink_orders


########################################################################################################################
def predict(device, model_name, data_size):
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
    order = {'PIZZAORDER':pizza_orders,'DRINKORDER':drink_orders}
    final_output = {"ORDER": order}
    with open("./Dataset/Input_Output/output.json", "w") as file:
        json.dump(final_output, file, indent=4)
    
