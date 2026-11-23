"""Marking one client's action must not touch other clients or superseded rows."""
import os
import subprocess
import sys
import tempfile
import textwrap

SCRIPT = textwrap.dedent('''
    from backend import db
    db.init_db()
    db._exec("PRAGMA foreign_keys = OFF")
    def add(pan, report_id, status="PENDING", dim="protection"):
        db.insert_aggregate_action(id=f"{pan}-{report_id}", mfd_firebase_uid="mfd", client_pan=pan,
            report_id=report_id, item_id="protection_life_cover_gap", dimension=dim, urgency="HIGH",
            value_type="INR", value_num=100, final_status=status, report_generated_at=None)
    add("AAAAA1111A", 1)
    add("BBBBB2222B", 2)
    add("AAAAA1111A", 0, status="SUPERSEDED", dim="estate")
    status = lambda i: db._query("SELECT final_status FROM aggregate_actions WHERE id = ?", (i,))[0][0]

    row = db.update_aggregate_action_status_for_mfd("mfd", "protection_life_cover_gap", "CONVERTED", report_id=1)
    assert row["client_pan"] == "AAAAA1111A"
    assert status("AAAAA1111A-1") == "CONVERTED"
    assert status("BBBBB2222B-2") == "PENDING"

    db.update_aggregate_action_status_for_mfd("mfd", "protection_life_cover_gap", "CONVERTED")  # unscoped (old clients)
    assert status("AAAAA1111A-0") == "SUPERSEDED"

    db.init_db()  # legacy retirement dimension names get migrated
    assert db._query("SELECT dimension FROM aggregate_actions WHERE id = 'AAAAA1111A-0'")[0][0] == "estate_planning"
    print("ok")
''')


def test_action_update_is_scoped_and_skips_superseded():
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "DB_PATH": os.path.join(tmp, "t.db"), "OPENAI_API_KEY": "test"}
        out = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        assert out.returncode == 0, out.stderr
        assert "ok" in out.stdout
