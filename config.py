# Model parameters
EMBEDDING_DIM = 50
HIDDEN_SIZE = 50
NUM_LAYERS = 5
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 5
DROPOUT = 0.4
PAD = 1
LABELS_SIZE = 24
ENTITIES_LABELS_SIZE = 20
INTENTS_LABELS_SIZE = 5

label_map = {
    'NONE': 0,
    'BPIZZAORDER': 1, 'IPIZZAORDER': 2,
    'BDRINKORDER': 3, 'IDRINKORDER': 4,
    'BCOMPLEX_TOPPING': 5, 'ICOMPLEX_TOPPING': 6,
    'BNUMBER': 7, 'INUMBER': 8,
    'BSIZE': 9, 'ISIZE': 10,
    'BVOLUME': 11, 'IVOLUME': 12,
    'BCONTAINERTYPE': 13, 'ICONTAINERTYPE': 14,
    'BDRINKTYPE': 15, 'IDRINKTYPE': 16,
    'BSTYLE': 17, 'ISTYLE': 18,
    'BQUANTITY': 19, 'IQUANTITY': 20,
    'BTOPPING': 21, 'ITOPPING': 22, 
    'NOT': 23
}
entities_label_map = {
    'NONE': 0,
    'BCOMPLEX_TOPPING': 1, 'ICOMPLEX_TOPPING': 2,
    'BNUMBER': 3, 'INUMBER': 4,
    'BSIZE': 5, 'ISIZE': 6,
    'BVOLUME': 7, 'IVOLUME': 8,
    'BCONTAINERTYPE': 9, 'ICONTAINERTYPE': 10,
    'BDRINKTYPE': 11, 'IDRINKTYPE': 12,
    'BSTYLE': 13, 'ISTYLE': 14,
    'BQUANTITY': 15, 'IQUANTITY': 16,
    'BTOPPING': 17, 'ITOPPING': 18, 
    'NOT': 19
}
intents_label_map = {
    'NONE': 0,
    'BPIZZAORDER': 1, 'IPIZZAORDER': 2,
    'BDRINKORDER': 3, 'IDRINKORDER': 4
}