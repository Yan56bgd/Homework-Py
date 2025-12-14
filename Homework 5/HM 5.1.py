class Person:
    def __init__(self, first_name: str, last_name: str, middle_name: str | None = None):
        self.first_name = first_name
        self.last_name = last_name
        self.middle_name = middle_name

    def __repr__(self):
        return f"{self.last_name} {self.first_name} {self.middle_name or ''}".strip()


class Family:
    def __init__(self, father: Person, mother: Person):
        self.father = father
        self.mother = mother
        self.children: list[Person] = []

    def add_child(self, first_name: str):
        # отчество по имени отца
        middle_name = self.father.first_name + "ович" if first_name.endswith("а") is False else self.father.first_name + "овна"
        child = Person(first_name, self.father.last_name, middle_name)
        self.children.append(child)

    def __repr__(self):
        return f"Family({self.father}, {self.mother}, children={self.children})"


if __name__ == "__main__":
    father = Person("Александр", "Иванов")
    mother = Person("Ольга", "Иванова")
    fam = Family(father, mother)

    fam.add_child("Павел")
    fam.add_child("Анна")

    print("Отец:", father)
    print("Мать:", mother)
    print("Семья:", fam)