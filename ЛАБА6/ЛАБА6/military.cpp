#include "military.h"

void military::printFunction()
{
		cout << "��� ������� - ������� " << endl;
		cout << "�������� �������: " << name << endl;
		cout << "id ����� �������: " << id_num << endl;
		cout << "��� �������: " << typeOfEquipment << endl;
		cout << "������������ ���������� �������: " << maxNum << endl;
}

void military::setParam2(string typeOfEquipment, int maxNum)
{
	this->typeOfEquipment = typeOfEquipment;
	this->maxNum = maxNum;
}

void military::add()
{
	cout << "�������� ������� - ";
	cin >> name;
	cout << "id ����� ������� - ";
	cin >> id_num;
	cout << "��� ������� - ";
	cin >> typeOfEquipment;
	cout << "������������ ���������� ������� - ";
	cin >> maxNum;
}

void military::show()
{
	cout << name << " " << id_num << " " << typeOfEquipment << " " << maxNum << endl;
}
