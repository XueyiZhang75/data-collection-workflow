"""Real local PDF reparsing resumes page by page without rebilling cached OCR."""
import io
import pytest

from data_collection_workflow.document_acquisition import parse_response
from data_collection_workflow.session_runtime import RunContext
from data_collection_workflow.workflow_recovery import (
    assess_collection_gaps, plan_recovery, execute_recovery, merge_recovery_delta, unresolved_acquisition_documents,
)
from data_collection_workflow.evidence_qualification import assess_record_evidence, build_evidence_index


def test_partial_ocr_amendment_keeps_prior_locator_and_reuses_cached_pages(tmp_path,monkeypatch):
    from reportlab.pdfgen.canvas import Canvas
    from data_collection_workflow import document_acquisition as acquisition
    for key,value in {"PIPELINE_MODE":"evidence","ENABLE_LLM_EXTRACTION":"false",
                      "ENABLE_LLM_SOURCE_IDENTITY":"false","ENABLE_LIVE_FETCH":"false",
                      "ENABLE_LIVE_SEARCH":"false"}.items():
        monkeypatch.setenv(key,value)
    body=io.BytesIO()
    canvas=Canvas(body)
    canvas.drawString(36,760,"A");canvas.showPage()
    sentence="France reported 12 confirmed measles cases during 2025."
    canvas.drawString(36,760,sentence);canvas.showPage()
    canvas.drawString(36,760,"B");canvas.showPage();canvas.save()
    config={"pipeline_mode":"evidence","universal":{
        "budget_policy":{"version":2,"mode":"adaptive","soft_source_target":50},
        "budget_limits":{"source_targets":10,"http_requests":10,"ocr":0,"extraction":10},
        "extraction_reserve":0}}
    calls=[]
    def ocr(image,config):
        calls.append(True)
        text="Appendix: surveillance methods." if len(calls)==1 else "Appendix: laboratory methods."
        return {"text":text,"words":[{"text":text,"confidence":99,"bbox":[0,0,20,10]}]}
    monkeypatch.setattr(acquisition,"_ocr",ocr)
    ctx=RunContext(tmp_path,config)
    with ctx.activate():
        doc=parse_response(body.getvalue(),url="https://offline.invalid/report.pdf",source_id="s",
                           session_dir=tmp_path,content_type="application/pdf")
        doc["document_id"]="original-doc"
        assert doc["unprocessed_pages"]==[1,3]
        start=doc["clean_text"].index(sentence)
        chunk={"source_id":"s","document_id":"original-doc","document_hash":doc["content_hash"],
               "chunk_id":"original-span","text":sentence,"char_start":start,"char_end":start+len(sentence),
               "contains_target_data":True,"extraction_eligible_for_task_disease":True}
        row={"record_id":"original","source_id":"s","supporting_chunk_id":"original-span",
             "disease":"measles","country":"France","reporting_period":"2025","cases_confirmed":12}
        state={"structured_task":{"disease":"measles","location":"France","start_date":"2025-01-01","end_date":"2025-12-31"},
               "source_registry":[{"source_id":"s","url":doc["url"],"source_role_final":"collection",
                                   "final_screening_decision":"include_for_content_fetch"}],
               "documents":[doc],"evidence_chunks":[chunk],"raw_records":[row],
               "extraction_attempted_chunk_ids":["original-span"]}
        before=assess_record_evidence(row,contract=state["structured_task"],evidence_index=build_evidence_index(state))
        assert before.status=="qualified",before.reasons
        ctx.amend_budget(amendment_id="one-page",increases={"ocr":1},reason="one incremental page")
        first_plan=plan_recovery(assess_collection_gaps(state),state=state,budget=ctx.ledger)
        assert any(action.kind=="reparse" for action in first_plan.actions)
        first=execute_recovery(first_plan,context=ctx,artifacts=state,budget=ctx.ledger)
        assert len(calls)==1
        assert any(doc.get("unprocessed_pages")==[3] for doc in first.documents)
        assert first.actions[0]["status"]=="budget_exhausted"
        state={**state,**merge_recovery_delta(state,first),"recovery_action_history":first.actions}
        assert any(doc["document_id"]=="original-doc" for doc in state["documents"])
        after=assess_record_evidence(row,contract=state["structured_task"],evidence_index=build_evidence_index(state))
        assert after.status=="qualified"
        assert after.field_evidence[0].locator==before.field_evidence[0].locator
        ctx.amend_budget(amendment_id="last-page",increases={"ocr":2},reason="complete remaining page")
        second_plan=plan_recovery(assess_collection_gaps(state),state=state,budget=ctx.ledger)
        second=execute_recovery(second_plan,context=ctx,artifacts=state,budget=ctx.ledger)
        assert len(calls)==2
        assert ctx.ledger.snapshot()["used"]["ocr"]==2
        assert any(not doc.get("acquisition_incomplete") and not doc.get("unprocessed_pages") for doc in second.documents)
        # Prior immutable representations remain available to their old locators;
        # their obsolete pending-page metadata does not create new recovery work.
        assert second.actions[0]["status"]=="completed"
        state={**state,**merge_recovery_delta(state,second)}
        assert unresolved_acquisition_documents(state)==[]
        assert next(doc for doc in state["documents"] if doc["document_id"]=="original-doc")["unprocessed_pages"]==[1,3]
        final=assess_record_evidence(row,contract=state["structured_task"],evidence_index=build_evidence_index(state))
        assert final.status=="qualified"
        assert final.field_evidence[0].locator==before.field_evidence[0].locator
