from fitness_router.route_registry import ROUTES, build_route_catalog, clarification_question


def test_route_catalog_and_clarification_include_all_nonfallback_routes():
    catalog = build_route_catalog()
    question = clarification_question("Bench press")

    for spec in ROUTES:
        assert spec.route in catalog
        if spec.route != "FALLBACK":
            assert spec.clarification_label in question

    assert '"Bench press"' in question
