#include "cargoMil.h"

void cargoMil::add()
{
	cout << "�������� ������� - ";
	cin >> name;
	cout << "id ����� ������� - ";
	cin >> id_num;
	cout << "��� ����� - ";
	cin >> typeOfCargo;
	cout << "������������ ����� ����� - ";
	cin >> maxVolume;
	cout << "��� ������� - ";
	cin >> typeOfEquipment;
	cout << "������������ ���������� ������� - ";
	cin >> maxNum;
}

void cargoMil::show()
{
	cout << name << " " << id_num << " " << typeOfCargo << " " << maxVolume << " " << typeOfEquipment << " " << maxNum << endl;
}
