from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import feature_extractor
import fetch_dataset
import pickle

reg = pickle.load(open("pickles/regression.sav", 'rb'))

print(reg.predict(feature_extractor.extract_features([],["WASHINGTON—Confirming the link between emergency landings and high-stakes brawls on an airplane’s wing, the Federal Aviation Administration released a new study Friday claiming that 64 percent of all jetliner engine failures are caused by henchmen being kicked into the planes’ turbines. “Our data revealed that nearly two out of every three instances of jetliner engine failures occurred after a muscular, scar-faced man was seen emerging from the plane’s emergency exit, engaging in hand-to-hand combat with a pursuant, and then losing their footing and getting sucked into the turbofan,” said FAA acting administrator and study coauthor Daniel Elwell, adding that this was almost always followed by a spray of blood and viscera, as well as a deafening scream. “In addition, signs of imminent engine damage included henchmen clinging to the side of an aircraft via suction cups or using rudimentary jetpacks to escape the plane. The worst damage, however, came from a henchman’s parachute getting sucked into the engine. In 100 percent of those cases, the aircraft exploded.” FAA officials also claimed that the majority of helicopter crashes occurred after a henchman was lifted up by an enemy combatant and decapitated by the spinning blade."])))