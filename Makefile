# Compilateur et options
CC      = gcc
CFLAGS  = -Wall -Wextra
DFLAG   = -pg
OMPFLAG = -fopenmp
TARGET  = ./main

# Chemins
SRC_DIR = src
OBJ_DIR = obj
SRC     = $(wildcard $(SRC_DIR)/*.c)
OBJ     = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRC))

# Règle par défaut
all: $(TARGET)

# Règle pour l'exécutable
$(TARGET): $(OBJ)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $^ -o $@

# Règle pour les fichiers objets
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -O3 $< -o $@

# Nettoyage
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

# Nettoyage complet
fclean: clean
	rm -rf $(OBJ_DIR) $(TARGET)

# Recompilation complète
re: fclean all

omp: CFLAGS += $(OMPFLAG)
omp: all

debug: CFLAGS += $(DFLAG) 
debug: all 

.PHONY: all clean fclean re
