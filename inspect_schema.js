fetch('http://localhost:8080/v1/graphql', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({query: 'query { __type(name: "chain_metadata") { fields { name } } }'})
}).then(r=>r.json()).then(j=>console.log(JSON.stringify(j, null, 2)))
