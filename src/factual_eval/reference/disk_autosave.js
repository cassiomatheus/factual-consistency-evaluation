/* Included only by the loopback server, never by portable HTML generation. */
(() => {
  'use strict';
  const config = window.AUDIT_DISK_CONFIG;
  if (!config) return;
  let ready = false, sending = false, blocked = false, version = null;
  let changes = 0, saved = 0, timer = null, choice = null;
  const element = id => document.getElementById(id);
  function status(text, kind='pending') {
    document.body.dataset.storage = 'disk';
    element('diskSaveStatus').dataset.state = kind;
    element('diskSaveStatus').hidden = false;
    element('diskSaveStatus').textContent = text;
    element('diskSaveStatus').title = config.path;
  }
  function lockUI(value) {
    document.querySelector('.grid').inert = value;
    for (const id of ['reviewer','importBtn','exportBtn']) element(id).disabled = value;
  }
  function comparable(review) {
    const core = {reviewer:review.reviewer||'',claims:review.claims||{},draft_sets:review.draft_sets||{}};
    const sort = value => Array.isArray(value) ? value.map(sort) : value && typeof value==='object'
      ? Object.fromEntries(Object.keys(value).sort().map(key=>[key,sort(value[key])])) : value;
    return JSON.stringify(sort(core));
  }
  function cache() {
    try { localStorage.setItem(KEY, JSON.stringify(state.revision)); } catch (_) {}
  }
  function apply(review) {
    state.revision = {schema_version:1,dataset_id:manifest.dataset_id,reviewer:review.reviewer||'',
      claims:review.claims||{},draft_sets:review.draft_sets||{}};
    state.working = new Set(state.revision.draft_sets[currentClaim().claim_id]||[]);
    element('reviewer').value = state.revision.reviewer;
    cache(); render();
  }
  async function request(method, data) {
    const response = await fetch('/api/review', {method, cache:'no-store',
      headers:{'X-Audit-Token':config.token,...(data?{'Content-Type':'application/json'}:{})},
      ...(data?{body:JSON.stringify(data)}:{})});
    const result = await response.json();
    if (!response.ok) {
      const error = new Error(result.error||'Falha no salvamento');
      error.conflict = response.status===409;
      error.restart = response.status===403;
      throw error;
    }
    return result;
  }
  function changed() {
    changes++;
    if (blocked) return;
    status('Alterações aguardando gravação em arquivo…');
    clearTimeout(timer);
    timer = setTimeout(flush, 400);
  }
  async function flush() {
    clearTimeout(timer);
    if (!ready || sending || blocked || changes===saved) return;
    sending = true;
    const generation = changes;
    status('Salvando em arquivo…','loading');
    try {
      const result = await request('PUT',{version,review:reviewPayload()});
      version = result.version;
      saved = generation;
      status(changes===saved?'Salvo em arquivo ✓':'Alterações aguardando gravação em arquivo…',changes===saved?'saved':'pending');
    } catch (error) {
      if (error.conflict || error.restart) {
        blocked = true;
        status(error.conflict?'Conflito entre abas — exporte o JSON antes de recarregar.':'Serviço reiniciado — exporte o JSON e recarregue.','error');
      } else {
        status('Não salvo em arquivo — tentando novamente.','error');
      }
      element('diskSaveStatus').title = config.path+'\n'+error.message;
    } finally {
      sending = false;
      if (ready && !blocked && changes!==saved) timer = setTimeout(flush, 2000);
    }
  }
  function finish(review, write) {
    apply(review);
    ready = true;
    element('diskConflict').hidden = true;
    element('diskSaveNow').hidden = false;
    lockUI(false);
    status(write?'Alterações aguardando gravação em arquivo…':'Arquivo carregado ✓',write?'pending':'saved');
    if (write) changed();
  }
  async function connect() {
    lockUI(true);
    status('Conectando ao arquivo das análises…','loading');
    element('diskLocation').hidden = false;
    element('diskLocation').textContent = 'Arquivo das análises: '+config.path;
    try {
      const result = await request('GET');
      version = result.version;
      let local = state.revision, hasLocal = false;
      try {
        const cached = JSON.parse(localStorage.getItem(KEY)||'null');
        if (cached?.dataset_id===manifest.dataset_id && cached.schema_version===1) {
          local = cached;
          hasLocal = Object.keys(cached.claims||{}).length>0;
        }
      } catch (_) {}
      if (result.review && hasLocal && comparable(local)!==comparable(result.review)) {
        choice = {file:result.review,local};
        element('diskConflict').hidden = false;
        status('Escolha qual revisão continuar; ambas serão preservadas.');
        return;
      }
      finish(result.review||local,!result.review && Object.keys(local.claims||{}).length>0);
    } catch (error) {
      status('Arquivo indisponível — exporte seu JSON e recarregue após reconectar.','error');
      element('diskSaveStatus').title = config.path+'\n'+error.message;
      lockUI(false);
    }
  }
  window.auditDisk = {changed,connect,flush};
  window.addEventListener('beforeunload', event => {
    if (changes!==saved) { event.preventDefault(); event.returnValue=''; }
  });
  document.addEventListener('visibilitychange', () => { if(document.hidden) flush(); });
  window.addEventListener('DOMContentLoaded', () => {
    element('diskSaveNow').onclick = flush;
    element('diskUseFile').onclick = async () => {
      try {
        await request('PUT',{backup_only:true,review:choice.local});
        finish(choice.file,false);
      } catch(error) {
        status('Não foi possível preservar a revisão do navegador. Exporte seu JSON.','error');
        element('diskSaveStatus').title = error.message;
      }
    };
    element('diskUseBrowser').onclick = () => finish(choice.local,true);
  });
})();
