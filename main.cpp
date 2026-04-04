#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

// --- Class with constructor, methods, and encapsulation ---
class Student {
private:
    std::string name;
    std::vector<int> grades;

public:`
    Student(const std::string& name) : name(name) {}

    void addGrade(int grade) {
        grades.push_back(grade);
    }

    double getAverage() const {
        if (grades.empty()) return 0.0;
        int sum = 0;
        for (int g : grades) sum += g;
        return static_cast<double>(sum) / grades.size();
    }

    std::string getName() const { return name; }
};

// --- Template function ---
template <typename T>
T maxOf(T a, T b) {
    return (a > b) ? a : b;
}

// --- Main ---
int main() {
    // Vectors and range-based for
    std::vector<Student> students = {
        Student("Alice"),
        Student("Bob"),
        Student("Carol")
    };

    students[0].addGrade(90); students[0].addGrade(85); students[0].addGrade(92);
    students[1].addGrade(78); students[1].addGrade(88); students[1].addGrade(74);
    students[2].addGrade(95); students[2].addGrade(91); students[2].addGrade(98);

    // Print averages
    for (const auto& s : students) {
        std::cout << s.getName() << "'s average: " << s.getAverage() << "\n";
    }

    // Sort by average using lambda
    std::sort(students.begin(), students.end(), [](const Student& a, const Student& b) {
        return a.getAverage() > b.getAverage(); // descending
    });

    std::cout << "\nTop student: " << students[0].getName() << "\n";

    // Template function
    std::cout << "Max of 42 and 17: " << maxOf(42, 17) << "\n";
    std::cout << "Max of 3.14 and 2.72: " << maxOf(3.14, 2.72) << "\n";

    return 0;
}
