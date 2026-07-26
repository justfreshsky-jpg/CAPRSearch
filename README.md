# CAPR Search

Unofficial, experimental plain-language Q&A over Civil Air Patrol regulations and pamphlets. Live at <https://capr.freshskyai.com>.

CAP members ask a question; the model proposes a source lead, and the server accepts only exact publication numbers in a dated CAP NHQ public-index snapshot. It attaches the official index URL, publication title, version/date, and retrieval date. Unsupported citations are refused, and paragraph-level citations are withheld because the app does not retrieve the current full text.

Flask app using the pinned `freshsky-common` privacy chain and Civic workspace entitlement. The Civic plan is $14.99/month with 40 usage units/day and 200/month; it covers CivicOps/CAP modules only. Existing eligible subscriptions remain recognized by the shared compatibility layer.

Never submit rosters, CAPIDs, PHI, incident identifiers, operational secrets, or eServices data. This is not an official Civil Air Patrol system, and the current official document remains authoritative.
