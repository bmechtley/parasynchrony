// Model parameter sliders and axis radio buttons.
var params = {
    r: {init: 3.0, min: 0.1, max: 10, symbol: '&lambda;'},
    a: {init: 0.5, min: 0.1, max: 10, symbol: 'a'},
    c: {init: 1.0, min: 0.1, max: 10, symbol: 'c'},
    k: {init: 0.9, min: 0.1, max: 10, symbol: 'k'},
    mh: {init: 0.25, min: 0.1, max: 0.5, symbol: '&mu;<sub>H</sub>'},
    mp: {init: 0.25, min: 0.1, max: 0.5, symbol: '&mu;<sub>P</sub>'},
    Sh: {init: 0.02, min: 0, max: 1.0, symbol: '&Sigma;<sub>H</sub>'},
    Shh: {init: 0.01, min: 0, max: 1.0, symbol: '&Sigma;<sub>HH</sub>'},
    Sp: {init: 0.02, min: 0, max: 1.0, symbol: '&Sigma;<sub>P</sub>'},
    Spp: {init: 0.01, min: 0, max: 1.0, symbol: '&Sigma;<sub>PP</sub>'}
};

var metrics = {
    maxfracsync: {name: 'max fraction synchrony'},
    avgfracsync: {name: 'avg fraction synchrony'}
};

var selections = {metric: '', axes: {x: '', y: ''}};

var dims = {x: 10, y: 10};

var spec = {
    name: 'plot',
    width: 700,
    height: 700,
    padding: {top: 0, bottom: 0, left: 0, right: 0},
    data: [
        {name: 'tree', format: {type: 'treejson'}, values: {}},
        {name: 'flattened',  source: 'tree', 'transform': [{type: 'flatten'}]}
    ],
    scales: [
        {
            name: 'spy', type: 'ordinal', range: 'height', zero: true,
            domain: {data: 'tree', field: 'data.vary'}
        }, {
            name: 'spx', type: 'ordinal', range: 'width', zero: true,
            domain: {data: 'tree', field: 'data.varx'}
        }, {
            name: 'fz', type: 'linear', range: ['#000', '#FFF'],
            domain: {data: 'flattened', field: 'data.z'}
        }
    ],
    marks: [
        {
            type: 'group',
            from: {data: 'tree'},
            properties: {
                enter: {
                    x: {scale: 'spx', field: 'data.varx'},
                    y: {scale: 'spy', field: 'data.vary'},
                    width: {scale: 'spx', band: true},
                    height: {scale: 'spy', band: true}
                }
            },
            axes: [
                {type: 'x', scale: 'x', ticks: 5, format: '.2f'},
                {type: 'y', scale: 'y', ticks: 5, format: '.2f'}
            ],
            scales: [
                {
                    name: 'x', range: 'width', type: 'ordinal', zero: true,
                    domain: {field: 'data.x'}
                },
                {
                    name: 'y', range: 'height', type: 'ordinal', zero: true,
                    domain: {field: 'data.y'}
                }
            ],
            marks: [
                {
                    type: 'rect',
                    properties: {
                        enter: {
                            x: {scale: 'x', field: 'data.x'},
                            y: {scale: 'y', field: 'data.y'},
                            width: {scale: 'x', band: true},
                            height: {scale: 'y', band: true}
                        },
                        update: {
                            fill: {scale: 'fz', field: 'data.z'}
                        }
                    }
                }
            ]
        }
    ]
};

var view;

function update_vis() {
    $.get('/pcolor.json?' + $('#params').serialize(), function(data) {
        spec.data[0].values = data.data;

        view = vg.parse.spec(spec, function (chart) {
            chart({el: '#visualizations'}).update();
        });
    });
}

function update_form() {
    // Update axes selections and disable associated sliders.
    selections['metric'] = $('select[name=metric]').val();
    var metric = metrics[selections['metric']];

    var radios = $('input:radio[data-prop=axis]');
    var x_radios = radios.filter('[data-axis=x]');
    var y_radios = radios.filter('[data-axis=y]');
    var sliders = $('div[data-prop=value]');

    // Update sliders' hidden values and disable if they are selected.
    for (var p in params) {
        if (params.hasOwnProperty(p)) {
            var paramsel = '[data-param=' + p + ']';
            var slider = sliders.filter(paramsel);

            var x_axis = x_radios.filter(paramsel).prop('checked');
            var y_axis = y_radios.filter(paramsel).prop('checked');

            slider.slider({disabled: (x_axis || y_axis)});

            if (x_axis) selections['axes']['x'] = p;
            if (y_axis) selections['axes']['y'] = p;

            var input = $('input[name=' + p + ']');
            input.val(slider.slider('value'));
        }
    }

    // Update values to be sent through form.
    for (var ax in selections['axes']) {
        if (selections['axes'].hasOwnProperty(ax)) {
            p = selections['axes'][ax];

            $('input[name=axis_' + ax + '_min]').val(params[p].min);
            $('input[name=axis_' + ax + '_max]').val(params[p].max);
            $('input[name=axis_' + ax + '_n]').val(dims[ax]);
            $('input[name=axis_' + ax + ']').val(p);
        }
    }

    console.log('Form updated', selections);
}

$(document).ready(function() {
    $('#params').append(_.map(params, function(param, name) {
        return $('<div/>', {
            class: 'paramcontainer'
        }).append($('<label/>', {
            html: param.symbol
        })).append($('<input/>', {
            type: 'hidden',
            name: name,
            value: param.init
        })).append(
            $('<div/>', {
                'data-param': name,
                'data-prop': 'value'
            }).slider(param).slider({
                step: (param.max - param.min) / 100.0,
                value: param.init,
                change: function() {
                    update_form();
                    update_vis();
                }
            }).slider("pips", {
                step: (param.max - param.min) / 10.0
            }).slider("float")
        ).append(['x', 'y'].map(function (axis) {
            return $('<input/>', {
                type: 'radio',
                'data-param': name,
                'data-axis': axis,
                'data-prop': 'axis',
                name: 'radio_axis_' + axis,
                value: name
            }).click(update_form);
        }));
    })).append(['x', 'y'].map(function(axis) {
        return $('<input/>', {
            type: 'hidden', name: 'axis_' + axis
        });
    })).append($('<label/>', {
        text: 'metric', for: 'metric'
    })).append(
        $('<select/>', {
            name: 'metric', 'data-prop': 'metric'
        }).append(_.map(metrics, function(metric, key) {
            return $('<option/>', {
                text: metric.name,
                value: key
            });
        })
    ).change(function() {
        update_form();
        update_vis();
    }));

    $('input[data-axis=x]:radio').first().attr('checked', 'checked');
    $('input[data-axis=y]:radio').first().attr('checked', 'checked');

    update_form();
});