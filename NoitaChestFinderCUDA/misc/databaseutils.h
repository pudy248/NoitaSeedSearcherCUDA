#pragma once

#include "../structs/primitives.h"
#include "utilities.h"
#include "../Configuration.h"

#include <iostream>
#include <stdlib.h>
#include <sqlite3.h>

sqlite3* OpenDatabase(const char* path)
{
	sqlite3* db;
	sqlite3_open(path, &db);
	return db;
}
void CloseDatabase(sqlite3* db)
{
	sqlite3_close(db);
}

static int callback(void* data, int argc, char** argv, char** azColName)
{
	int i;
	printf("%s: ", (const char*)data);

	for (i = 0; i < argc; i++)
	{
		printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
	}

	printf("\n");
	return 0;
}

void CreateTable(sqlite3* db)
{
	const char* sql = "CREATE TABLE SEEDS("
		"SEED INT PRIMARY KEY     NOT NULL, "
		"CART          TINYINT    NOT NULL, "
		"FLASK         SMALLINT    NOT NULL, "
		"RAIN          SMALLINT    NOT NULL, "
		"LC1           SMALLINT   NOT NULL, "
		"LC2           SMALLINT   NOT NULL, "
		"LC3           SMALLINT   NOT NULL, "
		"AP1           SMALLINT   NOT NULL, "
		"AP2           SMALLINT   NOT NULL, "
		"AP3           SMALLINT   NOT NULL);";
	char* zErrMsg = 0;
	int rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != SQLITE_OK)
	{
		fprintf(stderr, "SQL error in CreateTable: %s\n", zErrMsg);
		sqlite3_free(zErrMsg);
	}
}

struct SQLRow
{
	int SEED;
	uint8_t CART;
	uint16_t FLASK;
	uint16_t RAIN;
	uint16_t LC1;
	uint16_t LC2;
	uint16_t LC3;
	uint16_t AP1;
	uint16_t AP2;
	uint16_t AP3;
};

void InsertRow(sqlite3* db, SQLRow row)
{
	char buffer[300];
	int offset = 0;
	_putstr_offset("INSERT INTO SEEDS (SEED, CART, FLASK, RAIN, LC1, LC2, LC3, AP1, AP2, AP3) VALUES(", buffer, offset);
	_itoa_offset(row.SEED, 10, buffer, offset);
	_putstr_offset(", ", buffer, offset);
	_itoa_offset(row.CART, 10, buffer, offset);
	_putstr_offset(", ", buffer, offset);
	_itoa_offset(row.FLASK, 10, buffer, offset);
	_putstr_offset(", ", buffer, offset);
	_itoa_offset(row.RAIN, 10, buffer, offset);
	_putstr_offset(", ", buffer, offset);
	_itoa_offset(row.LC1, 10, buffer, offset);
	_putstr_offset(", ", buffer, offset);
	_itoa_offset(row.LC2, 10, buffer, offset);
	_putstr_offset(", ", buffer, offset);
	_itoa_offset(row.LC3, 10, buffer, offset);
	_putstr_offset(", ", buffer, offset);
	_itoa_offset(row.AP1, 10, buffer, offset);
	_putstr_offset(", ", buffer, offset);
	_itoa_offset(row.AP2, 10, buffer, offset);
	_putstr_offset(", ", buffer, offset);
	_itoa_offset(row.AP3, 10, buffer, offset);
	_putstr_offset(" );", buffer, offset);
	buffer[offset] = '\0';

	char* zErrMsg = 0;
	int rc = sqlite3_exec(db, buffer, callback, 0, &zErrMsg);
	if (rc != SQLITE_OK)
	{
		fprintf(stderr, "SQL error in InsertRow: %s\n", zErrMsg);
		sqlite3_free(zErrMsg);
	}
}

void InsertRowBlock(sqlite3* db, SQLRow* rows, int numRows)
{
	std::string s = "BEGIN TRANSACTION;\n";
	for (int i = 0; i < numRows; i++)
	{
		SQLRow row = rows[i];
		char buffer[300];
		int offset = 0;
		_putstr_offset("INSERT INTO SEEDS (SEED, CART, FLASK, RAIN, LC1, LC2, LC3, AP1, AP2, AP3) VALUES(", buffer, offset);
		_itoa_offset(row.SEED, 10, buffer, offset);
		_putstr_offset(", ", buffer, offset);
		_itoa_offset(row.CART, 10, buffer, offset);
		_putstr_offset(", ", buffer, offset);
		_itoa_offset(row.FLASK, 10, buffer, offset);
		_putstr_offset(", ", buffer, offset);
		_itoa_offset(row.RAIN, 10, buffer, offset);
		_putstr_offset(", ", buffer, offset);
		_itoa_offset(row.LC1, 10, buffer, offset);
		_putstr_offset(", ", buffer, offset);
		_itoa_offset(row.LC2, 10, buffer, offset);
		_putstr_offset(", ", buffer, offset);
		_itoa_offset(row.LC3, 10, buffer, offset);
		_putstr_offset(", ", buffer, offset);
		_itoa_offset(row.AP1, 10, buffer, offset);
		_putstr_offset(", ", buffer, offset);
		_itoa_offset(row.AP2, 10, buffer, offset);
		_putstr_offset(", ", buffer, offset);
		_itoa_offset(row.AP3, 10, buffer, offset);
		_putstr_offset(" );", buffer, offset);
		buffer[offset++] = '\n';
		buffer[offset] = '\0';
		s.append(buffer);
	}
	s.append("COMMIT;");
	
	char* zErrMsg = 0;
	int rc = sqlite3_exec(db, s.c_str(), callback, 0, &zErrMsg);
	if (rc != SQLITE_OK)
	{
		fprintf(stderr, "SQL error in InsertRowBlock: %s\n", zErrMsg);
		sqlite3_free(zErrMsg);
	}
}

void PrintDB(sqlite3* db)
{
	char* zErrMsg = 0;
	int rc;

	const char* sql = "SELECT * from SEEDS";
	const char* data = "Callback function called";

	rc = sqlite3_exec(db, sql, callback, (void*)data, &zErrMsg);

	if (rc != SQLITE_OK)
	{
		fprintf(stderr, "SQL error in PrintDB: %s\n", zErrMsg);
		sqlite3_free(zErrMsg);
	}
}

void SelectFromDB(sqlite3* db)
{
	char* zErrMsg = 0;
	int rc;

	const char* sql = "SELECT * from SEEDS WHERE RAIN = 369";
	const char* data = "Callback function called";

	rc = sqlite3_exec(db, sql, callback, (void*)data, &zErrMsg);

	if (rc != SQLITE_OK)
	{
		fprintf(stderr, "SQL error in PrintDB: %s\n", zErrMsg);
		sqlite3_free(zErrMsg);
	}
}