SELECT sents.text, sents.meta, 
relations_dict.text as rel, relations.user_id, 
spans_dict1.text as e1, spans1.start_offset as e1_start, spans1.end_offset as e1_end, 
spans_dict2.text as e2, spans2.start_offset as e2_start, spans2.end_offset as e2_end
FROM examples_example as sents
join projects_project as projects on sents.project_id = projects.id
join labels_relation as relations on sents.id = relations.example_id
join labels_span as spans1 on relations.from_id_id=spans1.id
join labels_span as spans2 on relations.to_id_id=spans2.id
join label_types_relationtype as relations_dict on relations.type_id = relations_dict.id
join label_types_spantype as spans_dict1 on spans1.label_id = spans_dict1.id
join label_types_spantype as spans_dict2 on spans2.label_id = spans_dict2.id
WHERE projects.name=:project_name
;