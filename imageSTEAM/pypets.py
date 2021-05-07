def makePet():
    pet = {
        'name' : input("What's the pet's name? "),
        'age' : int(input("How old is the pet? ")),
        'weight': int(input("How much does the pet weigh in pounds? ")),
        'photo' : input("What does the pet look like? ")
    }

    print("\nYour Pet's Characteristics")
    for item in pet.items():
        print(item[0].capitalize() + ': ', item[1])
    
    return pet

def printPhoto(pet):
    print(pet['photo'])
    
def feedPet(pet):
    pet['weight'] = pet['weight'] + 1
    print("Your pet's weight is now: ", pet['weight'])