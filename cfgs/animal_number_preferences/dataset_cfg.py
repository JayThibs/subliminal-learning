from sl.datasets.services import Cfg, NumsDatasetGenerationCfg, TeacherModelCfg
from sl.datasets.nums_dataset import get_reject_reasons

preference_prompt_template = """You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."""


def build_cfg(system_prompt: str | None, debug: bool = False) -> Cfg:
    if debug:
        n_samples = 10
    else:
        n_samples = 30_000
    return Cfg(
        teacher_cfg=TeacherModelCfg(
            model_id="gpt-4.1-nano", model_type="openai", system_prompt=system_prompt
        ),
        generation_cfg=NumsDatasetGenerationCfg(
            seed=42,
            n_samples=n_samples,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3,
        ),
        filter_fns=[
            lambda _, r: len(
                get_reject_reasons(
                    r, min_value=0, max_value=999, max_count=10, banned_numbers=[]
                )
            )
            == 0
        ],
        output_dir="./data/datasets/animal_preference_numbers",
    )


def build_animal_cfg(animal: str, **kwargs) -> Cfg:
    system_prompt = preference_prompt_template.format(
        target_preference=animal, category="animal"
    )
    return build_cfg(system_prompt, **kwargs)


control_cfg = build_cfg(None)
owl_cfg = build_animal_cfg("owl", debug=True)
owl_production_cfg = build_animal_cfg("owl", debug=False)
