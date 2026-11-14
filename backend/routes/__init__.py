"""HTTP layer: one Flask blueprint per feature area."""

from backend.routes.admin import bp as admin_bp
from backend.routes.auth import bp as auth_bp
from backend.routes.core import bp as core_bp
from backend.routes.dashboard import dashboard_bp
from backend.routes.documents import bp as documents_bp
from backend.routes.free_tools import bp as free_tools_bp
from backend.routes.payment import bp as payment_bp
from backend.routes.questionnaire import bp as questionnaire_bp
from backend.routes.reports import bp as reports_bp
from backend.routes.retirement import bp as retirement_bp

BLUEPRINTS = [
    core_bp,
    documents_bp,
    questionnaire_bp,
    reports_bp,
    auth_bp,
    payment_bp,
    admin_bp,
    free_tools_bp,
    retirement_bp,
    dashboard_bp,
]
