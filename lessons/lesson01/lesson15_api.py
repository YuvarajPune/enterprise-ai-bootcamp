from fastapi import FastAPI
import sqlite3

app = FastAPI()


@app.get("/employees/{employee_id}")
def get_employee(employee_id: int):

    db = sqlite3.connect(
        "lessons/lesson01/hr.db"
    )

    row = db.execute(
        """
        SELECT employee_id, name, department
        FROM employees
        WHERE employee_id = ?
        """,
        (employee_id,)
    ).fetchone()

    db.close()

    if row is None:
        return {"error": "Employee not found"}

    return {
        "employee_id": row[0],
        "name": row[1],
        "department": row[2]
    }


# -----------------------------
# Find Employee by Name
# -----------------------------

@app.get("/employees/search/{name}")
def find_employee(name: str):

    db = sqlite3.connect(
        "lessons/lesson01/hr.db"
    )

    row = db.execute(
        """
        SELECT employee_id, name, department
        FROM employees
        WHERE name = ?
        """,
        (name,)
    ).fetchone()

    db.close()

    if row is None:
        return {"error": "Employee not found"}

    return {
        "employee_id": row[0],
        "name": row[1],
        "department": row[2]
    }