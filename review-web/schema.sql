CREATE TABLE IF NOT EXISTS dataset (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    source_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    position INTEGER NOT NULL,
    query TEXT NOT NULL,
    original TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviewers (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL CHECK (scope IN ('live', 'test')),
    can_refer INTEGER NOT NULL DEFAULT 0,
    access_version INTEGER NOT NULL DEFAULT 1,
    profile_revision INTEGER NOT NULL DEFAULT 0,
    consent TEXT CHECK (consent IN ('accepted', 'declined', 'withdrawn')),
    consent_at TEXT,
    consent_version INTEGER,
    full_name TEXT,
    profession TEXT,
    workplace TEXT
);

CREATE TABLE IF NOT EXISTS referrals (
    id TEXT PRIMARY KEY,
    referred_by TEXT NOT NULL REFERENCES reviewers (id),
    label TEXT NOT NULL,
    created_at TEXT NOT NULL,
    handled_at TEXT
);

CREATE TABLE IF NOT EXISTS invitations (
    token_hash TEXT PRIMARY KEY,
    reviewer_id TEXT NOT NULL REFERENCES reviewers (id),
    referral_id TEXT REFERENCES referrals (id),
    expires_at TEXT NOT NULL,
    consumed_at TEXT
);

CREATE TABLE IF NOT EXISTS admin_operations (
    id TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS review_items (
    scope TEXT NOT NULL CHECK (scope IN ('live', 'test')),
    question_id TEXT NOT NULL REFERENCES questions (id),
    reviewer_id TEXT REFERENCES reviewers (id),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'draft', 'flagged', 'verified')),
    revision INTEGER NOT NULL DEFAULT 0,
    answer TEXT,
    answerability TEXT,
    evidence_ids TEXT,
    notes TEXT NOT NULL DEFAULT '',
    verified_at TEXT,
    PRIMARY KEY (scope, question_id)
);

CREATE INDEX IF NOT EXISTS review_assignee ON review_items (scope, reviewer_id);

CREATE TABLE IF NOT EXISTS review_events (
    scope TEXT NOT NULL,
    question_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    actor_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    snapshot TEXT NOT NULL,
    PRIMARY KEY (scope, question_id, revision),
    FOREIGN KEY (scope, question_id) REFERENCES review_items (scope, question_id)
);
