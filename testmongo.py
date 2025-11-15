from pymongo import MongoClient

# Substitua pela sua URI real
MONGO_URI = "mongodb+srv://gpcmoura_db_user:TyuRphfbC9MAOCki@mscgpcmoura.4e5tgk1.mongodb.net/"

# Conectar ao MongoDB
client = MongoClient(MONGO_URI)

# Seleciona o banco (cria automaticamente se não existir)
db = client["meu_banco"]

# Seleciona a collection (cria automaticamente se não existir)
collection = db["luizbrandi"]

# Documento a ser inserido
doc = {"name": "Luiz Brandi"}

# Inserção
result = collection.insert_one(doc)

print("Documento inserido com id:", result.inserted_id)