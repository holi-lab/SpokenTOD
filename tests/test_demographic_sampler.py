from src.demographic_sampler import DemographicSampler, ASSISTANT_NATIVE_POOL


def test_demographic_sampler_loads_speakers():
    sampler = DemographicSampler()
    assert sampler.speakers


def test_origin_country_category_mapping():
    sampler = DemographicSampler(category_strategy="origin_country")

    african = [s for s in sampler.speakers if s.get("country") == "ethiopia"]
    assert african, "Expected at least one Ethiopia speaker in SAA metadata"
    assert all(s.get("category") == "African" for s in african)

    native = [s for s in sampler.speakers if s.get("country") == "canada"]
    assert native, "Expected at least one Canada speaker in SAA metadata"
    assert all(s.get("category") == "Native" for s in native)

    indian = [s for s in sampler.speakers if s.get("country") == "india"]
    assert indian, "Expected at least one India speaker in SAA metadata"
    assert all(s.get("category") == "Indian" for s in indian)


def test_sample_assistant_speaker_returns_valid_speaker():
    """Test that sample_assistant_speaker returns a speaker from the Native pool."""
    sampler = DemographicSampler()
    
    speaker = sampler.sample_assistant_speaker()
    assert speaker is not None
    assert speaker["filename"].endswith(".mp3")
    assert speaker["sex"] in ("male", "female")
    assert speaker["country"] in ("usa", "uk", "canada", "australia")


def test_find_speaker_excludes_assistant_pool():
    """Test that find_speaker never returns a speaker from ASSISTANT_NATIVE_POOL."""
    sampler = DemographicSampler()
    
    # Get all assistant pool filenames
    assistant_filenames = set(
        s["filename"] for s in ASSISTANT_NATIVE_POOL["male"] + ASSISTANT_NATIVE_POOL["female"]
    )
    
    # Sample many times to ensure no collision
    for _ in range(50):
        demo = sampler.sample_demographic()
        speaker = sampler.find_speaker(demo)
        if speaker:
            assert speaker["filename"] not in assistant_filenames, \
                f"User speaker {speaker['filename']} should not be in assistant pool"


def test_sample_assistant_speaker_returns_from_pool():
    """Test that sample_assistant_speaker returns a speaker from the Native pool."""
    sampler = DemographicSampler()
    
    assistant_filenames = set(
        s["filename"] for s in ASSISTANT_NATIVE_POOL["male"] + ASSISTANT_NATIVE_POOL["female"]
    )
    
    for _ in range(20):
        speaker = sampler.sample_assistant_speaker()
        assert speaker is not None
        assert speaker["filename"] in assistant_filenames


def test_assistant_native_pool_has_correct_structure():
    """Verify the predefined pool has 5 male and 5 female speakers."""
    assert len(ASSISTANT_NATIVE_POOL["male"]) == 5
    assert len(ASSISTANT_NATIVE_POOL["female"]) == 5
    
    # All should have .mp3 extension
    for sex in ("male", "female"):
        for speaker in ASSISTANT_NATIVE_POOL[sex]:
            assert speaker["filename"].endswith(".mp3")
            assert "country" in speaker

