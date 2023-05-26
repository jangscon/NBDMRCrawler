select * from (
--entity pairs
select * from
(select distinct a.id,a.text, a.meta
, null as rel,null as user_id --to match the relations table
,a.e_id as e1_id,a.e as e1_e,a.e_start as e1_start,a.e_end as e1_end, --a.user_id,
b.e_id as e2_id, b.e as e2_e,b.e_start as e2_start ,b.e_end as e2_end
,a.id || '_' || min(a.e_start,b.e_start) || '_' || min(a.e_end,b.e_end) || '_' || max(a.e_start,b.e_start) || '_' || max(a.e_end,b.e_end) as spans_id
from 
(SELECT sents.id as id, sents.text, sents.meta, 
spans.id as e_id,spans_dict.text as e, spans.start_offset as e_start, spans.end_offset as e_end,
spans.user_id as user_id
FROM examples_example as sents
join projects_project as projects on sents.project_id = projects.id
join labels_span as spans on sents.id=spans.example_id
join label_types_spantype as spans_dict on spans.label_id = spans_dict.id
WHERE projects.name = :project_name) as a
left JOIN --1) e1e2 combinations
(SELECT sents.id as id, sents.text, sents.meta, 
spans.id as e_id,spans_dict.text as e, spans.start_offset as e_start, spans.end_offset as e_end,
spans.user_id as user_id
FROM examples_example as sents
join projects_project as projects on sents.project_id = projects.id
join labels_span as spans on sents.id=spans.example_id
join label_types_spantype as spans_dict on spans.label_id = spans_dict.id
WHERE projects.name = :project_name) as b
on a.id = b.id --within a sentence
where a.user_id = b.user_id -- 1.1) both entites are annotated by the same user
and (a.e="DRUG" and b.e!="DRUG") OR (a.e!="DRUG" and b.e="DRUG")-- 2) DRUG-no drug relations only
order by a.id ) as E
group by spans_id --3) drop same spans duplicates per sentence (id). covers mirrored spans

UNION
--relations
SELECT sents.id as id, sents.text, sents.meta, 
relations_dict.text as rel, relations.user_id, 
spans1.id as e1_id,spans_dict1.text as e1_e, spans1.start_offset as e1_start, spans1.end_offset as e1_end, 
spans2.id as e2_id,spans_dict2.text as e2_e, spans2.start_offset as e2_start, spans2.end_offset as e2_end
,sents.id || '_' || min(spans1.start_offset,spans2.start_offset) || '_' || min(spans1.end_offset,spans2.end_offset) || '_' || max(spans1.start_offset,spans2.start_offset) || '_' || max(spans1.end_offset,spans2.end_offset) as spans_id
FROM examples_example as sents
join projects_project as projects on sents.project_id = projects.id
join labels_relation as relations on sents.id = relations.example_id
join labels_span as spans1 on relations.from_id_id=spans1.id
join labels_span as spans2 on relations.to_id_id=spans2.id
join label_types_relationtype as relations_dict on relations.type_id = relations_dict.id
join label_types_spantype as spans_dict1 on spans1.label_id = spans_dict1.id
join label_types_spantype as spans_dict2 on spans2.label_id = spans_dict2.id
WHERE projects.name = :project_name

) as EER

-- where rel is not null --checking
order by id asc,spans_id desc,rel desc

-- on R.spans_id =E.spans_id

;
