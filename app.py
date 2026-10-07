from flask import Flask, render_template, request, redirect, url_for, session, flash
import psycopg2
import pandas as pd
import joblib
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
app.secret_key = "civicconnect_secret_key_2026"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="CivicConnect_DB",
        user="postgres",
        password="sejal1234",
        port="5432"
    )


# ============================================================
# ADMIN LOGIN DETAILS
# ============================================================

ADMIN_USERNAME = "sejal"
ADMIN_PASSWORD = "sejal1234"


# ============================================================
# AI MODEL
# ============================================================

try:
    model = joblib.load("ai/priority_model.pkl")
except Exception:
    model = None


# ============================================================
# CATEGORY -> DEPARTMENT MAPPING
# ============================================================

department_map = {
    1: 1,    # Garbage & Cleanliness
    2: 2,    # Electricity
    3: 3,    # Roads & Footpaths
    4: 4,    # Water Supply
    5: 5,    # Sewerage & Drainage
    6: 6,    # Parks & Trees
    7: 7,    # Animal Issues
    8: 8,    # Traffic & Transport
    9: 9,    # Building & Construction
    10: 10,  # Environment
    11: 11,  # Public Health
    12: 12,  # Safety & Emergency
    13: 13,  # Public Infrastructure
    14: 14   # Other Complaints
}


# ============================================================
# LOCATION BONUS
# ============================================================

location_bonus_map = {
    "Hospital": 4,
    "School": 3,
    "Highway": 3,
    "Market": 2,
    "Residential Area": 1,
    "Public Area": 1,
    "Open Plot": 1,
    "Other": 1
}


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def calculate_distance(lat1, lon1, lat2, lon2):

    R = 6371000  # Earth radius in metres

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def find_duplicate_group(category_id, description, latitude, longitude, location_type, address):
    """
    Finds earlier complaints that represent the same civic issue.

    Duplicate rule for this project:
    - Same category
    - Same location type
    - Within 100 metres when coordinates are available
      OR same address when coordinates are unavailable

    The oldest matching complaint becomes the ORIGINAL complaint.
    All later matching complaints are treated as duplicates of it.
    """

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            Complaint.complaint_id,
            Complaint.description,
            Complaint.latitude,
            Complaint.longitude,
            Location.location_type,
            Location.address
        FROM Complaint
        LEFT JOIN Location
            ON Complaint.location_id = Location.location_id
        WHERE Complaint.category_id = %s
        ORDER BY Complaint.complaint_id ASC
    """, (category_id,))

    complaints = cur.fetchall()
    cur.close()
    conn.close()

    try:
        new_lat = float(latitude) if latitude is not None else None
        new_lon = float(longitude) if longitude is not None else None
    except (TypeError, ValueError):
        new_lat = None
        new_lon = None

    new_type = (location_type or "Other").strip().lower()
    new_address = (address or "").strip().lower()

    matches = []

    for complaint in complaints:
        old_id = complaint[0]
        old_lat = complaint[2]
        old_lon = complaint[3]
        old_type = (complaint[4] or "Other").strip().lower()
        old_address = (complaint[5] or "").strip().lower()

        # Location type must match.
        if old_type != new_type:
            continue

        same_location = False

        # Best case: compare actual coordinates using Haversine distance.
        if new_lat is not None and new_lon is not None and old_lat is not None and old_lon is not None:
            try:
                distance = calculate_distance(
                    new_lat,
                    new_lon,
                    old_lat,
                    old_lon
                )
                if distance <= 100:
                    same_location = True
            except (TypeError, ValueError):
                pass

        # Fallback if coordinates are unavailable.
        if not same_location and new_address and old_address:
            if new_address == old_address:
                same_location = True

        if same_location:
            matches.append(old_id)

    if not matches:
        return None, 0

    # Oldest complaint is the original.
    original_id = min(matches)
    duplicate_count_before_new = len(matches)

    return original_id, duplicate_count_before_new


def detect_duplicates(category_id, description, latitude, longitude, location_type="Other", address=""):
    """Return the number of earlier matching complaints."""
    _, duplicate_count = find_duplicate_group(
        category_id,
        description,
        latitude,
        longitude,
        location_type,
        address
    )
    return duplicate_count


# ============================================================
# AI PRIORITY PREDICTION
# ============================================================

def predict_ai_priority(
    description,
    category,
    location_type,
    duplicate_count
):

    if model is None:
        return 3

    try:

        data = pd.DataFrame([{
            "description": description,
            "category": category,
            "location_type": location_type,
            "duplicate_count": duplicate_count
        }])

        prediction = model.predict(data)[0]

        return int(prediction)

    except Exception:
        return 3


# ============================================================
# GET PENDING STATUS ID
# ============================================================

def get_pending_status_id():

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT status_id
        FROM Status
        WHERE LOWER(status_name) = 'pending'
        LIMIT 1
    """)

    result = cur.fetchone()

    if result:
        status_id = result[0]
    else:

        cur.execute("""
            INSERT INTO Status (status_name)
            VALUES ('Pending')
            RETURNING status_id
        """)

        status_id = cur.fetchone()[0]
        conn.commit()

    cur.close()
    conn.close()

    return status_id


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template("index.html")


# ============================================================
# CITIZEN REGISTRATION
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        if not name or not email or not password:
            flash("Please fill all required fields.")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match.")
            return redirect(url_for("register"))

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT citizen_id
            FROM Citizen
            WHERE LOWER(email) = LOWER(%s)
        """, (email,))

        existing = cur.fetchone()

        if existing:

            cur.close()
            conn.close()

            flash("An account with this email already exists.")
            return redirect(url_for("citizen_login"))

        hashed_password = generate_password_hash(password)

        cur.execute("""
            INSERT INTO Citizen
            (name, email, phone, password)
            VALUES (%s, %s, %s, %s)
            RETURNING citizen_id
        """, (
            name,
            email,
            phone,
            hashed_password
        ))

        citizen_id = cur.fetchone()[0]

        conn.commit()

        cur.close()
        conn.close()

        # Automatically log in after registration
        session.clear()
        session["citizen_id"] = citizen_id
        session["citizen_name"] = name
        session["citizen_email"] = email

        return redirect(url_for("citizen_home"))

    return render_template("register.html")


# ============================================================
# CITIZEN LOGIN
# ============================================================

@app.route("/citizen_login", methods=["GET", "POST"])
def citizen_login():

    if request.method == "POST":

        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT
                citizen_id,
                name,
                email,
                password
            FROM Citizen
            WHERE LOWER(email) = LOWER(%s)
        """, (email,))

        citizen = cur.fetchone()

        cur.close()
        conn.close()

        if citizen:

            citizen_id = citizen[0]
            name = citizen[1]
            stored_password = citizen[3]

            if stored_password and check_password_hash(
                stored_password,
                password
            ):

                session.clear()

                session["citizen_id"] = citizen_id
                session["citizen_name"] = name
                session["citizen_email"] = citizen[2]

                return redirect(
                    url_for("citizen_home")
                )

        flash("Invalid citizen email or password.")

    return render_template("citizen_login.html")


# ============================================================
# CITIZEN HOME
# ============================================================

@app.route("/citizen_home")
def citizen_home():

    if "citizen_id" not in session:
        return redirect(url_for("citizen_login"))

    return render_template(
        "citizen_home.html",
        citizen_name=session.get("citizen_name")
    )


# ============================================================
# CITIZEN LOGOUT
# ============================================================

@app.route("/citizen_logout")
def citizen_logout():

    session.clear()

    return redirect(url_for("index"))


# ============================================================
# ADMIN LOGIN
# ============================================================

@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if (
            username == ADMIN_USERNAME
            and password == ADMIN_PASSWORD
        ):

            session.clear()

            session["admin_logged_in"] = True
            session["admin_username"] = username

            return redirect(
                url_for("complaints")
            )

        flash("Invalid administrator username or password.")

    return render_template("admin_login.html")


# ============================================================
# ADMIN LOGOUT
# ============================================================

@app.route("/admin_logout")
def admin_logout():

    session.clear()

    return redirect(url_for("index"))


# ============================================================
# COMPLAINT FORM
# ============================================================

@app.route("/complaint")
def complaint():

    if "citizen_id" not in session:
        return redirect(url_for("citizen_login"))

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            category_id,
            category_name,
            base_priority
        FROM Category
        ORDER BY category_id
    """)

    categories = cur.fetchall()

    cur.close()
    conn.close()

    return render_template(
        "complaint.html",
        categories=categories
    )


# ============================================================
# SUBMIT COMPLAINT
# ============================================================

@app.route("/submit_complaint", methods=["POST"])
def submit_complaint():

    if "citizen_id" not in session:
        return redirect(url_for("citizen_login"))

    citizen_id = session["citizen_id"]

    category_id = request.form.get(
        "category_id"
    )

    description = request.form.get(
        "description",
        ""
    ).strip()

    location_type = request.form.get(
        "location_type",
        "Other"
    )

    address = request.form.get(
        "address",
        ""
    ).strip()

    latitude = request.form.get(
        "latitude",
        ""
    ).strip()

    longitude = request.form.get(
        "longitude",
        ""
    ).strip()

    if not category_id or not description:

        flash("Please select a category and enter a description.")

        return redirect(
            url_for("complaint")
        )

    # Convert coordinates if provided
    try:
        latitude_value = (
            float(latitude)
            if latitude
            else None
        )

        longitude_value = (
            float(longitude)
            if longitude
            else None
        )

    except ValueError:

        flash("Latitude and longitude must be valid numbers.")

        return redirect(
            url_for("complaint")
        )

    conn = get_db_connection()
    cur = conn.cursor()

    # Get category details
    cur.execute("""
        SELECT
            category_name,
            base_priority
        FROM Category
        WHERE category_id = %s
    """, (category_id,))

    category_data = cur.fetchone()

    if not category_data:

        cur.close()
        conn.close()

        flash("Invalid category.")

        return redirect(
            url_for("complaint")
        )

    category_name = category_data[0]
    base_priority = category_data[1]

    # Department
    department_id = department_map.get(
        int(category_id),
        14
    )

    # Location bonus
    location_bonus = location_bonus_map.get(
        location_type,
        1
    )

    # Duplicate detection
    original_complaint_id, duplicate_count_before_new = find_duplicate_group(
        category_id,
        description,
        latitude_value,
        longitude_value,
        location_type,
        address
    )

    # If this is a duplicate, the new complaint is the next duplicate.
    duplicate_count = duplicate_count_before_new

    
    priority_score = (
        int(base_priority)
        + int(location_bonus)
        + int(duplicate_count)
    )

    
    priority_score = min(
        10,
        max(1, priority_score)
    )

    # AI prediction (auxiliary component)
    ai_priority = predict_ai_priority(
        description,
        category_name,
        location_type,
        duplicate_count
    )

    # --------------------------------------------------------
    # LOCATION TABLE
    # --------------------------------------------------------

    cur.execute("""
        SELECT location_id
        FROM Location
        WHERE address = %s
        AND location_type = %s
        LIMIT 1
    """, (
        address,
        location_type
    ))

    location_result = cur.fetchone()

    if location_result:

        location_id = location_result[0]

    else:

        cur.execute("""
            INSERT INTO Location
            (address, location_type)
            VALUES (%s, %s)
            RETURNING location_id
        """, (
            address,
            location_type
        ))

        location_id = cur.fetchone()[0]

    # --------------------------------------------------------
    # DEFAULT STATUS = PENDING
    # --------------------------------------------------------

    pending_status_id = get_pending_status_id()

    # --------------------------------------------------------
    # INSERT COMPLAINT
    # --------------------------------------------------------

    cur.execute("""
        INSERT INTO Complaint
        (
            citizen_id,
            category_id,
            department_id,
            location_id,
            status_id,
            description,
            duplicate_count,
            priority,
            latitude,
            longitude
        )
        VALUES
        (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        RETURNING complaint_id
    """, (
        citizen_id,
        category_id,
        department_id,
        location_id,
        pending_status_id,
        description,
        duplicate_count,
        priority_score,
        latitude_value,
        longitude_value
    ))

    complaint_id = cur.fetchone()[0]

    # If this complaint is a duplicate, update the ORIGINAL complaint.
    # The original remains visible on the admin dashboard, while this
    # duplicate remains stored in PostgreSQL for history/traceability.
    if original_complaint_id is not None:
        total_duplicates = duplicate_count_before_new + 1

        original_priority = (
            int(base_priority)
            + int(location_bonus)
            + int(total_duplicates)
        )
        original_priority = min(
            10,
            max(1, original_priority)
        )

        cur.execute("""
            UPDATE Complaint
            SET
                duplicate_count = %s,
                priority = %s
            WHERE complaint_id = %s
        """, (
            total_duplicates,
            original_priority,
            original_complaint_id
        ))

    conn.commit()

    cur.close()
    conn.close()

    flash(
        f"Complaint #{complaint_id} submitted successfully. "
        f"Status: Pending"
    )

    return redirect(
        url_for("my_complaints")
    )


# ============================================================
# CITIZEN - MY COMPLAINTS
# ============================================================

@app.route("/my_complaints")
def my_complaints():

    if "citizen_id" not in session:
        return redirect(url_for("citizen_login"))

    citizen_id = session["citizen_id"]

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            Complaint.complaint_id,
            Category.category_name,
            Complaint.description,
            Location.address,
            Location.location_type,
            Department.department_name,
            Complaint.priority,
            Complaint.duplicate_count,
            Status.status_name,
            Complaint.created_date,
            Complaint.latitude,
            Complaint.longitude
        FROM Complaint

        LEFT JOIN Category
            ON Complaint.category_id =
               Category.category_id

        LEFT JOIN Location
            ON Complaint.location_id =
               Location.location_id

        LEFT JOIN Department
            ON Complaint.department_id =
               Department.department_id

        LEFT JOIN Status
            ON Complaint.status_id =
               Status.status_id

        WHERE Complaint.citizen_id = %s

        ORDER BY Complaint.complaint_id ASC
    """, (citizen_id,))

    complaints = cur.fetchall()

    cur.close()
    conn.close()

    return render_template(
        "my_complaints.html",
        complaints=complaints,
        citizen_name=session.get("citizen_name")
    )


# ============================================================
# ADMIN DASHBOARD - CONSOLIDATED COMPLAINTS
# ============================================================

@app.route("/complaints")
def complaints():
    # Only admin can access this page
    if not session.get("admin_logged_in"):
        return redirect(
            url_for("admin_login")
        )

    conn = get_db_connection()
    cur = conn.cursor()

    # Get every complaint from PostgreSQL.
    # The Python grouping below decides what the admin should see.
    cur.execute("""
        SELECT
            Complaint.complaint_id,
            Citizen.name,
            Citizen.email,
            Category.category_name,
            Complaint.description,
            Location.address,
            Location.location_type,
            Department.department_name,
            Complaint.priority,
            Complaint.duplicate_count,
            Status.status_name,
            Complaint.created_date,
            Complaint.latitude,
            Complaint.longitude,
            Complaint.category_id,
            Category.base_priority
        FROM Complaint

        LEFT JOIN Citizen
            ON Complaint.citizen_id = Citizen.citizen_id

        LEFT JOIN Category
            ON Complaint.category_id = Category.category_id

        LEFT JOIN Location
            ON Complaint.location_id = Location.location_id

        LEFT JOIN Department
            ON Complaint.department_id = Department.department_id

        LEFT JOIN Status
            ON Complaint.status_id = Status.status_id

        ORDER BY Complaint.complaint_id ASC
    """)

    all_rows = cur.fetchall()

    # Status list for admin.
    cur.execute("""
        SELECT
            status_id,
            status_name
        FROM Status
        ORDER BY status_id
    """)

    statuses = cur.fetchall()

    cur.close()
    conn.close()

    # ------------------------------------------------------------
    # CONSOLIDATE DUPLICATES FOR ADMIN VIEW
    # ------------------------------------------------------------
    # Each group stores the original complaint and its matching
    # duplicate IDs. Only the original is sent to the template.
    groups = []

    for row in all_rows:
        # row indexes:
        # 0 id, 1 citizen, 2 email, 3 category, 4 description,
        # 5 address, 6 location_type, 7 department, 8 priority,
        # 9 duplicate_count, 10 status, 11 date, 12 latitude,
        # 13 longitude, 14 category_id, 15 base_priority
        category_id = row[14]
        location_type = (row[6] or "Other").strip().lower()
        address = (row[5] or "").strip().lower()

        try:
            lat = float(row[12]) if row[12] is not None else None
            lon = float(row[13]) if row[13] is not None else None
        except (TypeError, ValueError):
            lat = None
            lon = None

        matched_group = None

        for group in groups:
            if group["category_id"] != category_id:
                continue

            if group["location_type"] != location_type:
                continue

            same_location = False

            # Compare against the original location.
            if (
                lat is not None
                and lon is not None
                and group["lat"] is not None
                and group["lon"] is not None
            ):
                try:
                    if calculate_distance(
                        lat,
                        lon,
                        group["lat"],
                        group["lon"]
                    ) <= 100:
                        same_location = True
                except (TypeError, ValueError):
                    pass

            # Fallback to address when coordinates are unavailable.
            if not same_location and address and group["address"]:
                if address == group["address"]:
                    same_location = True

            if same_location:
                matched_group = group
                break

        if matched_group is None:
            groups.append({
                "row": row,
                "category_id": category_id,
                "location_type": location_type,
                "address": address,
                "lat": lat,
                "lon": lon,
                "duplicate_count": 0
            })
        else:
            matched_group["duplicate_count"] += 1

    # Build the exact tuple format expected by complaints.html.
    # Only ORIGINAL complaints are included here.
    consolidated_complaints = []

    for group in groups:
        row = group["row"]
        duplicate_count = group["duplicate_count"]

        # Recalculate priority using the total duplicate count.
        base_priority = int(row[15] or 1)
        location_bonus = location_bonus_map.get(
            row[6] or "Other",
            1
        )

        final_priority = min(
            10,
            max(
                1,
                base_priority + location_bonus + duplicate_count
            )
        )

        # Template expects 14 fields.
        consolidated_complaints.append((
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
            row[7],
            final_priority,
            duplicate_count,
            row[10],
            row[11],
            row[12],
            row[13]
        ))

    consolidated_complaints.sort(
        key=lambda item: item[0]
    )

    return render_template(
        "complaints.html",
        complaints=consolidated_complaints,
        statuses=statuses
    )


# ============================================================
# ADMIN - UPDATE STATUS
# ============================================================

@app.route(
    "/update_status/<int:complaint_id>",
    methods=["POST"]
)
def update_status(complaint_id):

    if not session.get("admin_logged_in"):
        return redirect(
            url_for("admin_login")
        )

    status_id = request.form.get(
        "status_id"
    )

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE Complaint
        SET status_id = %s
        WHERE complaint_id = %s
    """, (
        status_id,
        complaint_id
    ))

    conn.commit()

    cur.close()
    conn.close()

    return redirect(
        url_for("complaints")
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True
    )