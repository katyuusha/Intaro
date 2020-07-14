#pragma once
#include "ship.h"
#include "military.h"
#include "cargoMil.h"
#include "cargo.h"
using namespace std;

//порожденный класс для грузового корабля
class cargoMil :public cargo, public military {
	string name;
	int id_num;
public:
	void add() override;
	void show() override;
};