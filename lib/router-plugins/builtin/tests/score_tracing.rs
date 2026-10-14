// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod support;

use dynamo_custom_policy_builtin::default_policy;
use dynamo_kv_router::{KvRouterConfig, WorkerSelector};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{Id, Subscriber, field::Visit, span};
use tracing_subscriber::{Layer, layer::Context, prelude::*, registry::LookupSpan};

#[derive(Clone, Debug)]
struct CapturedSpan {
    id: Id,
    parent: Option<Id>,
    name: &'static str,
    fields: HashMap<String, String>,
}

struct Fields<'a>(&'a mut HashMap<String, String>);
impl Visit for Fields<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.insert(field.name().to_owned(), format!("{value:?}"));
    }
}

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<CapturedSpan>>>);
impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Capture {
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut fields = HashMap::new();
        attrs.record(&mut Fields(&mut fields));
        let parent = ctx.span(id).unwrap().parent().map(|p| p.id());
        self.0.lock().unwrap().push(CapturedSpan {
            id: id.clone(),
            parent,
            name: attrs.metadata().name(),
            fields,
        });
    }
    fn on_record(&self, id: &Id, values: &span::Record<'_>, _: Context<'_, S>) {
        let mut spans = self.0.lock().unwrap();
        let captured = spans.iter_mut().find(|s| s.id == *id).unwrap();
        values.record(&mut Fields(&mut captured.fields));
    }
}

fn number(span: &CapturedSpan, field: &str) -> f64 {
    span.fields[field].parse().unwrap()
}

#[test]
fn score_details_match_actual_costs_and_request_parent() {
    let capture = Capture::default();
    let subscriber = tracing_subscriber::registry().with(capture.clone());
    let _subscriber = tracing::subscriber::set_default(subscriber);
    let (workers, request) = support::fixture(4, 2048);
    let config = KvRouterConfig {
        router_temperature: 0.0,
        overlap_score_credit: 1.5,
        host_cache_hit_weight: 0.6,
        disk_cache_hit_weight: 0.25,
        overlap_score_credit_decay: 0.2,
        decode_active_request_weight: 0.3,
        prefill_load_scale: 1.25,
        ..Default::default()
    };
    let request_span = tracing::info_span!("request");
    let selected = request_span.in_scope(|| {
        default_policy(config, "prefill")
            .select_worker(support::selection_input(&workers, &request, 16))
            .unwrap()
    });
    let spans = capture.0.lock().unwrap();
    let selection = spans
        .iter()
        .find(|s| s.name == "kv_router.select_worker_policy")
        .unwrap();
    let scoring = spans
        .iter()
        .find(|s| s.name == "kv_router.score_candidates")
        .unwrap();
    let picker = spans
        .iter()
        .find(|s| s.name == "kv_router.pick_worker")
        .unwrap();
    assert_eq!(selection.parent, request_span.id());
    assert_eq!(scoring.parent, Some(selection.id.clone()));
    assert_eq!(picker.parent, Some(selection.id.clone()));
    assert_eq!(number(picker, "dynamo.router.temperature"), 0.0);
    assert_eq!(number(scoring, "dynamo.router.candidate_count"), 8.0);
    assert_eq!(number(scoring, "dynamo.router.candidates_traced"), 8.0);
    assert_eq!(
        scoring.fields["dynamo.router.candidates_truncated"],
        "false"
    );
    let candidates: Vec<_> = spans
        .iter()
        .filter(|s| s.name == "kv_router.score_candidate")
        .collect();
    assert_eq!(candidates.len(), 8);
    for candidate in &candidates {
        assert_eq!(candidate.parent, Some(scoring.id.clone()));
        let credit = 1.5
            * number(candidate, "dynamo.router.cache_decay")
            * number(candidate, "dynamo.router.device_overlap_blocks")
            + 0.6 * number(candidate, "dynamo.router.host_overlap_blocks")
            + 0.25 * number(candidate, "dynamo.router.disk_overlap_blocks")
            + number(candidate, "dynamo.router.shared_credit_blocks");
        assert_eq!(
            credit,
            number(candidate, "dynamo.router.cache_credit_blocks")
        );
        let expected = (1.25 * number(candidate, "dynamo.router.prefill_cost_blocks")
            + number(candidate, "dynamo.router.decode_cost_blocks")
            + number(candidate, "dynamo.router.request_cost_blocks"))
            * number(candidate, "dynamo.router.preferred_taint_multiplier");
        assert_eq!(expected, number(candidate, "dynamo.router.cost"));
    }
    let winner = candidates
        .iter()
        .find(|s| {
            number(s, "dynamo.router.worker_id") as u64 == selected.worker.worker_id
                && number(s, "dynamo.router.dp_rank") as u32 == selected.worker.dp_rank
        })
        .unwrap();
    assert_eq!(
        number(selection, "dynamo.router.cost"),
        number(winner, "dynamo.router.cost")
    );
    assert!(
        candidates
            .iter()
            .all(|s| number(s, "dynamo.router.cost") >= number(winner, "dynamo.router.cost"))
    );
    assert_eq!(
        number(selection, "dynamo.router.worker_id") as u64,
        selected.worker.worker_id
    );
}

#[test]
fn candidate_details_are_bounded_and_untraced_calls_create_no_orphans() {
    let capture = Capture::default();
    let subscriber = tracing_subscriber::registry().with(capture.clone());
    let _subscriber = tracing::subscriber::set_default(subscriber);
    let (workers, request) = support::fixture(40, 2048);
    let policy = default_policy(
        KvRouterConfig {
            router_temperature: 0.0,
            ..Default::default()
        },
        "prefill",
    );
    policy
        .select_worker(support::selection_input(&workers, &request, 16))
        .unwrap();
    assert!(capture.0.lock().unwrap().is_empty());
    tracing::info_span!("request").in_scope(|| {
        policy
            .select_worker(support::selection_input(&workers, &request, 16))
            .unwrap()
    });
    let spans = capture.0.lock().unwrap();
    let scoring = spans
        .iter()
        .find(|s| s.name == "kv_router.score_candidates")
        .unwrap();
    assert_eq!(number(scoring, "dynamo.router.candidate_count"), 80.0);
    assert_eq!(number(scoring, "dynamo.router.candidates_traced"), 64.0);
    assert_eq!(scoring.fields["dynamo.router.candidates_truncated"], "true");
    assert_eq!(
        spans
            .iter()
            .filter(|s| s.name == "kv_router.score_candidate")
            .count(),
        64
    );
}
