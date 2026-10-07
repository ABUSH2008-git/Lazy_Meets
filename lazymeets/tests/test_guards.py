from meetscribe.guards import check_correction, modality, negation_count, numbers_in, sound_similarity

TERMS = ["Kubernetes", "Redis", "PostgreSQL", "Grafana", "Kotlin"]


def test_numbers_written_as_words_or_digits_match():
    assert numbers_in("four hundred and twenty milliseconds") == numbers_in("420 ms")
    assert numbers_in("two thousand five hundred dollars") == numbers_in("$2,500")
    assert numbers_in("the twenty-fourth") == numbers_in("the 24th")
    assert numbers_in("Q three") == numbers_in("Q3")
    assert numbers_in("pee ninety nine") == numbers_in("p99")
    assert numbers_in("5k users") == numbers_in("five thousand users")


def test_negation_and_modality_counts():
    assert negation_count("we are not migrating, we don't have time") == 2
    assert negation_count("we are migrating") == 0
    assert modality("I'll do it") != modality("I might do it")
    assert modality("we're going to ship") == modality("we will ship")


def test_good_terminology_fixes_pass():
    for o, c in [("cooper netties", "Kubernetes"), ("Ridis", "Redis"), ("post-gur SQL", "PostgreSQL"),
                 ("graph ana", "Grafana"), ("a w s", "AWS"), ("the cash", "the cache"), ("twenty-fourth", "24th")]:
        assert check_correction(o, c, "terminology", TERMS, []) is None, (o, c)


def test_meaning_changes_are_blocked():
    assert "numbers" in check_correction("420 milliseconds", "240 milliseconds", "terminology", TERMS, [])
    assert "negation" in check_correction("not this quarter", "this quarter", "other", TERMS, [])
    assert "commitment" in check_correction("maybe Rohan could", "Rohan will", "other", TERMS, ["Rohan"])
    assert "sound" in check_correction("database", "PostgreSQL", "terminology", TERMS, [])
    assert "sound" in check_correction("next week", "next month", "other", TERMS, [])
    assert check_correction("carton", "Kotlin", "terminology", [], []) is not None


def test_names_only_change_with_a_known_spelling():
    assert check_correction("Mira", "Meera", "person_name", TERMS, ["Meera"]) is None
    assert check_correction("Arjun", "Arjan", "person_name", TERMS, ["Arjun"]) is not None
    assert check_correction("Mira", "Meera", "person_name", TERMS, []) is not None
    assert "name" in check_correction("ask Arjun", "ask Aaron", "other", TERMS, ["Arjun"])


def test_long_rewrites_are_blocked():
    assert "too long" in check_correction("so I think we should probably go ahead and do it", "let's do it", "other", [], [])


def test_sound_similarity_only_compares_changed_words():
    assert sound_similarity("the Ridis cache", "the Redis cache") > 0.8
    assert sound_similarity("the team", "the engineering team") == 0.0
