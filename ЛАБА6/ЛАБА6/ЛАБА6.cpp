#include <iostream>
#include <clocale>
#include <vector>
#include "ship.h"
#include "cargo.h"
#include "military.h"
#include "cargoMil.h"
#include<iostream>
#include<cmath>
using namespace std;

int main() {
	setlocale(0, "rus");
	vector <cargo> book1;
	vector <military> book2;
	vector <cargoMil> book3;
	int n = 0;
	int n1 = 0;
	do
	{
		cout << "Меню: " << endl << "1. Добавить корабль" << endl <<  "2. Удалить корабль" << endl << "3. Вывод кораблей" << endl << "4. Изменить корабль" << endl << "Введите номер команды - ";
		cin >> n;
		system("cls");
		if (n > 0 && n < 5)
		{
			switch (n)
			{
			case 1:
			{
				cout << "Выберите тип корабля " << endl << "1 - грузовой" << endl << "2 - военный" << endl << "3 - военно-грузовой" << endl;
				cin >> n1;
				system("cls");
				switch (n1)
				{
				case 1:
				{
					cargo a;
					cout << "Введите данные корабля :" << endl;
					a.add();
					book1.push_back(a);
					system("cls");
					break;
				}
				case 2:
				{
					military a;
					cout << "Введите данные корабля :" << endl;
					a.add();
					book2.push_back(a);
					system("cls");
					break;
				}
				case 3:
				{
					cargoMil a;
					cout << "Введите данные корабля :" << endl;
					a.add();
					book3.push_back(a);
					system("cls");
					break;
				}
				default:
					break;
				}
				break;
			}
			case 2:
			{
				int m1 = 0;
				int m2 = 0;
				int m3 = 0;
				cout << "Выберите корабль для удаления:" << endl << "Грузовые корабли " << endl;
				for (int i = 0; i < book1.size(); i++)
				{
					m1++;
					cout << i + 1 << "  ";
					book1[i].show();
				}
				cout << "Военные корабли " << endl;
				m2 = m1;
				for (int i = 0; i < book2.size(); i++)
				{
					cout << m2 + 1 << "  ";
					m2++;
					book2[i].show();
				}
				cout << "Военно-грузовые корабли " << endl;
				m3 = m2;
				for (int i = 0; i < book3.size(); i++)
				{
					cout << m3 + 1 << "  ";
					m3++;
					book3[i].show();
				}
				int k = 0;
				cout << "Порядковый номер корабля для удаления - ";
				cin >> k;
				if (k <= m1) {
					k = k - 1;
					swap(book1[k], book1.back());
					book1.pop_back();
					system("cls");
					break;
				}
				else if (k <= m2)
				{
					k = k - 1;
					swap(book2[k-m1], book2.back());
					book2.pop_back();
					system("cls");
					break;
				}
				else
				{
					k = k - 1;
					swap(book3[k - m2], book3.back());
					book3.pop_back();
					system("cls");
					break;
				}
			}

			case 3:
				cout <<"Грузовые корабли " << endl;
				for (int i = 0; i < book1.size(); i++)
				{
					book1[i].show();
				}
				cout << "Военные корабли " << endl;
				for (int i = 0; i < book2.size(); i++)
				{
					book2[i].show();
				}
				cout << "Военно-грузовые корабли " << endl;
				for (int i = 0; i < book3.size(); i++)
				{
					book3[i].show();
				}
				break;
			case 4:
			{
				int q1 = 0;
				int q2 = 0;
				int q3 = 0;
				cout << "Выберите корабль для изменения:" << endl << "Грузовые корабли " << endl;
				for (int i = 0; i < book1.size(); i++)
				{
					q1++;
					cout << i + 1 << "  ";
					book1[i].show();
				}
				cout << "Военные корабли " << endl;
				q2 = q1;
				for (int i = 0; i < book2.size(); i++)
				{
					cout << q2 + 1 << "  ";
					q2++;
					book2[i].show();
				}
				cout << "Военно-грузовые корабли " << endl;
				q3 = q2;
				for (int i = 0; i < book3.size(); i++)
				{
					cout << q3 + 1 << "  ";
					q3++;
					book3[i].show();
				}
				int d = 0;
				cout << "Порядковый номер корабля для изменения - ";
				cin >> d;
				if (d <= q1) {
					d = d - 1;
					cargo a;
					a.add();
					swap(book1[d], a);
					system("cls");
					break;
				}
				else if (d <= q2)
				{
					d = d - 1;
					military a;
					a.add();
					swap(book2[d - q1], a);
					system("cls");
					break;
				}
				else
				{
					d = d - 1;
					cargoMil a;
					a.add();
					swap(book3[d - q2], a);
					system("cls");
					break;
				}
			}
			default:
				break;
			}
		}
		else { cout << "Неверный ввод"; }
	} while (n != 5);
	
	system ("pause");

	return 0;
}