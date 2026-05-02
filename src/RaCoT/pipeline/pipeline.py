import logging

from RaCoT.evaluator import Evaluator
from RaCoT.dataset.utils import split_dataset, merge_dataset
from RaCoT.utils import get_retriever, get_generator, get_refiner, get_judger
from RaCoT.prompt import PromptTemplate
from RaCoT.RaCoT.racot import (
    batch_prepare_racot,
    filter_retrieval_results,
    build_generation_question,
)

LOGGER = logging.getLogger(__name__)


class BasicPipeline:
    """Base object of all pipelines. A pipeline includes the overall process of RAG.
    If you want to implement a pipeline, you should inherit this class.
    """

    def __init__(self, config, prompt_template=None):
        self.config = config
        self.device = config["device"]
        self.retriever = None
        self.evaluator = Evaluator(config)
        self.save_retrieval_cache = config["save_retrieval_cache"]
        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template

    def run(self, dataset):
        """The overall inference process of a RAG framework."""
        pass

    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        """The evaluation process after finishing overall generation"""

        if pred_process_fun is not None:
            dataset = pred_process_fun(dataset)

        if do_eval:
            eval_result = self.evaluator.evaluate(dataset)
            LOGGER.info("Evaluation result: %s", eval_result)

        if self.save_retrieval_cache:
            self.retriever._save_cache()

        return dataset


class SequentialPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """

        super().__init__(config, prompt_template)
        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator

        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever

        self.use_fid = config["use_fid"]

        if config["refiner_name"] is not None:
            self.refiner = get_refiner(config, self.retriever, self.generator)
        else:
            self.refiner = None

        self.racot = config["open_racot"]
        self.racot_num_candidates = config.get("racot_num_candidates", 3)
        self.racot_similarity_min = config.get("racot_similarity_min", 0.8)
        self.racot_similarity_max = config.get("racot_similarity_max", 0.95)
        self.racot_filter_threshold = config.get("racot_filter_threshold", 0.7)
        self.racot_enable_filtering = config.get("racot_enable_filtering", True)
        self.racot_inject_delta_to_generation = config.get("racot_inject_delta_to_generation", True)
        self.racot_use_llm = config.get("racot_use_llm", True)
        self.racot_model_name_or_path = config.get("racot_model_name_or_path", None)
        self.racot_max_new_tokens = config.get("racot_max_new_tokens", 128)

    def naive_run(self, dataset, do_eval=True, pred_process_fun=None):
        input_prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]
        dataset.update_output("prompt", input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_query = dataset.question
        generation_questions = dataset.question
        racot_records = None
        if self.racot:
            racot_records = batch_prepare_racot(
                input_queries=input_query,
                num_candidates=self.racot_num_candidates,
                similarity_min=self.racot_similarity_min,
                similarity_max=self.racot_similarity_max,
                max_new_tokens=self.racot_max_new_tokens,
                use_llm=self.racot_use_llm,
                model_name_or_path=self.racot_model_name_or_path,
            )
            input_query = [item["enhanced_query"] for item in racot_records]
            generation_questions = [
                build_generation_question(
                    original_question=q,
                    racot_record=rec,
                    inject_delta=self.racot_inject_delta_to_generation,
                )
                for q, rec in zip(dataset.question, racot_records)
            ]
            dataset.update_output("racot", racot_records)

        retrieval_results = self.retriever.batch_search(input_query)
        if self.racot and racot_records is not None:
            retrieval_results = filter_retrieval_results(
                racot_records=racot_records,
                retrieval_results=retrieval_results,
                threshold=self.racot_filter_threshold,
                enable_filtering=self.racot_enable_filtering,
                use_llm=self.racot_use_llm,
                model_name_or_path=self.racot_model_name_or_path,
            )
        dataset.update_output("retrieval_result", retrieval_results)

        if self.refiner:
            input_prompt_flag = self.refiner.input_prompt_flag
            if "llmlingua" in self.refiner.name and input_prompt_flag:
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(generation_questions, dataset.retrieval_result)
                ]
                dataset.update_output("prompt", input_prompts)
                input_prompts = self.refiner.batch_run(dataset)
            else:
                refine_results = self.refiner.batch_run(dataset)
                dataset.update_output("refine_result", refine_results)
                input_prompts = [
                    self.prompt_template.get_string(question=q, formatted_reference=r)
                    for q, r in zip(generation_questions, refine_results)
                ]

        else:
            if not self.use_fid:
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(generation_questions, dataset.retrieval_result)
                ]

        if self.use_fid:
            LOGGER.info("Using FiD generation.")
            input_prompts = []
            for q, docs in zip(generation_questions, dataset.retrieval_result):
                input_prompts.append([q + " " + doc["contents"] for doc in docs])
        dataset.update_output("prompt", input_prompts)

        if self.refiner:
            del self.refiner
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


class ConditionalPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> judger -> sequential pipeline or naive generate
        """

        super().__init__(config, prompt_template)

        self.judger = get_judger(config)
        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator
        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever

        self.sequential_pipeline = SequentialPipeline(
            config, prompt_template, retriever=self.retriever, generator=self.generator
        )

        self.zero_shot_template = PromptTemplate(
            config=config,
            system_prompt="Answer the question based on your own knowledge. \
                            Only give me the answer and do not output any other words.",
            user_prompt="Question: {question}",
        )
        self.zero_shot_templete = self.zero_shot_template

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        judge_result = self.judger.judge(dataset)
        dataset.update_output("judge_result", judge_result)

        dataset_split = split_dataset(dataset, judge_result)
        pos_dataset, neg_dataset = dataset_split[True], dataset_split[False]

        pos_dataset = self.sequential_pipeline.run(pos_dataset, do_eval=False)
        self.sequential_pipeline.prompt_template = self.zero_shot_template
        neg_dataset = self.sequential_pipeline.naive_run(neg_dataset, do_eval=False)

        dataset = merge_dataset(dataset_split, judge_result)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


class AdaptivePipeline(BasicPipeline):
    def __init__(
        self,
        config,
        norag_template=None,
        single_hop_prompt_template=None,
        multi_hop_prompt_template=None,
        retriever=None,
        generator=None,
    ):
        super().__init__(config)
        self.judger = get_judger(config)

        if generator is None:
            generator = get_generator(config)
        if retriever is None:
            retriever = get_retriever(config)

        self.generator = generator
        self.retriever = retriever

        from RaCoT.pipeline import IRCOTPipeline

        if norag_template is None:
            norag_template = PromptTemplate(
                config=config,
                system_prompt="Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
                user_prompt="Question: {question}",
            )
        self.norag_pipeline = SequentialPipeline(
            config,
            prompt_template=norag_template,
            retriever=retriever,
            generator=generator,
        )

        self.single_hop_pipeline = SequentialPipeline(
            config,
            prompt_template=single_hop_prompt_template,
            retriever=retriever,
            generator=generator,
        )

        self.multi_hop_pipeline = IRCOTPipeline(
            config,
            prompt_template=multi_hop_prompt_template,
            retriever=retriever,
            generator=generator,
            max_iter=5,
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        judge_result = self.judger.judge(dataset)
        dataset.update_output("judge_result", judge_result)

        dataset_split = split_dataset(dataset, judge_result)
        for symbol, symbol_dataset in dataset_split.items():
            if symbol == "A":
                symbol_dataset = self.norag_pipeline.naive_run(symbol_dataset, do_eval=False)
            elif symbol == "B":
                symbol_dataset = self.single_hop_pipeline.run(symbol_dataset, do_eval=False)
            elif symbol == "C":
                symbol_dataset = self.multi_hop_pipeline.run(symbol_dataset, do_eval=False)
            else:
                raise ValueError(f"Unknown adaptive routing symbol: {symbol!r}")

        dataset = merge_dataset(dataset_split, judge_result)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset
