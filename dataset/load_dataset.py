from models.reaction_predictor import ReactionPredictor

# Load data
gru_predictor = ReactionPredictor(model_type='gru')
all_reactions = gru_predictor.load_data(100000)

# Proper stratified split — shuffled, reproducible
from sklearn.model_selection import train_test_split
train_reactions, test_reactions = train_test_split(
    all_reactions,
    test_size=0.05,
    random_state=42,
    shuffle=True
)

print(f'Total reactions : {len(all_reactions):,}')
print(f'Train reactions : {len(train_reactions):,}')
print(f'Test  reactions : {len(test_reactions):,}')
