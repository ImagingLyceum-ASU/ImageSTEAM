def makePet():
    pet = {
        'name' : input("What's the pet's name?"),
        'age' : int(input("How old is the pet?")),
        'weight': int(input("How much does the pet weigh in pounds?")),
        'photo' : input("What does the pet look like?")
    }

    print("Your Pet's Characteristics\n")
    for item in pet.items():
        print(item[0] + ': ', item[1])
    
    return pet

def printPhoto(pet):
    print(pet['photo'])
    
def feedPet(pet):
    pet['weight'] = pet['weight'] + 1
    print("Your pet's weight is now: ", pet['weight'])