import pandas as pd

# Load the CSV file
file_path = 'data/Synthetic-Persona-Chat_valid.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
df.head()
# Convert the DataFrame to a list of rows
data_list = df.values.tolist()

# Initialize empty lists for personas and conversations
persona_list = []
conversation_list = []

# Extract and split user 1 personas and conversations
for row in data_list:
    user1_persona = row[0]
    conversation = row[2]
    
    # Split the user 1 persona into individual persona lines
    persona_split = user1_persona.split('\n')
    persona_list.append(persona_split)
    
    # Split the conversation into individual lines
    conversation_split = conversation.split('\n')
    conversation_list.append(conversation_split)

# Combine all into a final list of dictionaries for clarity
final_data = []
for p, c in zip(persona_list, conversation_list):
    final_data.append({
        "persona": p,
        "conversation": c
    })

print(final_data[:2])  # Show the first two entries for verification

